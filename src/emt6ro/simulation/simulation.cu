#include <emt6ro/common/grid.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include "emt6ro/common/debug.h"
#include "emt6ro/diffusion/grid-diffusion.h"
#include "emt6ro/diffusion/new-diffusion.h"
#include "emt6ro/diffusion/old-diffusion.h"
#include "emt6ro/division/cell-division.h"
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/statistics/statistics.h"
#include "simulation.h"

namespace emt6ro {

namespace detail {

__global__ void populateGridViewsKernel(GridView<Site> *views, uint32_t batch_size,
                                        Dims dims, Site *origin) {
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < batch_size) {
    views[idx].data = origin + dims.vol() * idx;
    views[idx].dims = dims;
  }
}

__host__ __device__ uint8_t vacantNeighbours(const GridView<Site> &grid, int32_t r, int32_t c) {
  return grid(r - 1, c - 1).isVacant() +
         grid(r - 1, c + 1).isVacant() +
         grid(r + 1, c - 1).isVacant() +
         grid(r + 1, c + 1).isVacant() +
         grid(r, c - 1).isVacant() +
         grid(r, c + 1).isVacant() +
         grid(r - 1, c).isVacant() +
         grid(r + 1, c).isVacant();
}

__global__ void cellSimulationKernel(GridView<Site> *grids, ROI *rois,
                                     Parameters *params, curandState_t *rand_states,
                                     bool *dividing_cells,
                                     Protocol *protocols, uint32_t step) {
  extern __shared__ bool division_ready[];
  uint8_t vacant_neighbours[4];
  const auto roi = rois[blockIdx.x];
  auto &grid = grids[blockIdx.x];
  const auto &protocol = protocols[blockIdx.x];
  const auto start_r = roi.origin.r;
  const auto start_c = roi.origin.c;
  const auto tid = blockDim.x * threadIdx.y + threadIdx.x;
  curandState_t *rand_state =
      rand_states + blockDim.x * blockDim.y * blockIdx.x + tid;
  CuRandEngine rand(rand_state);
  uint8_t subi = 0;
  GRID_FOR(start_r, start_c, roi.dims.height + start_r, roi.dims.width + start_c) {
    vacant_neighbours[subi] = vacantNeighbours(grid, r, c);
    ++subi;
  }
  __syncthreads();
  division_ready[tid] = false;
  subi = 0;
  GRID_FOR(start_r, start_c, roi.dims.height + start_r, roi.dims.width + start_c) {
    auto &site = grid(r, c);
    if (site.isOccupied()) {
      const auto vn = vacant_neighbours[subi];
      uint8_t alive = site.cell.updateState(site.substrates, *params, vn);
      if (alive) {
        auto dose = protocol.getDose(step);
        if (dose > 0) site.cell.irradiate(dose, params->cell_repair);
        site.cell.metabolise(site.substrates, params->metabolism);
        bool cycle_changed = site.cell.progressClock(params->time_step);
        alive = site.cell.tryRepair(params->cell_repair, cycle_changed, params->time_step, rand);
        if (alive) {
          division_ready[tid] = division_ready[tid] || site.cell.phase == Cell::CyclePhase::D;
        } else {
          site.state = Site::State::VACANT;
        }
      } else {
        site.state = Site::State::VACANT;
      }
    }
    ++subi;
  }
  __syncthreads();
  int t = tid;
  for (int d = 1; d < blockDim.y * blockDim.x; d *= 2) {
    if (t % 2 == 0) {
      division_ready[tid] = division_ready[tid] || division_ready[tid + d];
      t /= 2;
    }
    __syncthreads();
  }
  if (tid == 0) dividing_cells[blockIdx.x] = division_ready[0];
}

__global__ void cellDivisionKernel(GridView<Site> *lattices, Parameters *params,
                                    const bool *division_ready, curandState_t *rand_states) {
  if (!division_ready[blockIdx.x]) return;
  curandState_t *rand_state =
      rand_states + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x;
  CuRandEngine rand(rand_state);
  auto &lattice = lattices[blockIdx.x];
  divideCells(lattice, *params, rand);
}

}  // namespace detail

void Simulation::populateLattices() {
  auto blocks = (batch_size + 1023) / 1024;
  auto block_size = (batch_size > 1024) ? 1024 : batch_size;
  detail::populateGridViewsKernel<<<blocks, block_size, 0, stream_>>>
    (lattices.data(), batch_size, dims, data.data());
  KERNEL_DEBUG("populate")
}

Simulation::Simulation(Dims dims, uint32_t batch_size, const Parameters &parameters, uint32_t seed)
    : batch_size(batch_size)
    , dims(dims)
    , params(parameters)
    , d_params(device::alloc_unique<Parameters>(1))
    , data(batch_size * dims.vol())
    , lattices(batch_size)
    , diffusion_tmp_data(batch_size * dims.vol())
    , vacant_neighbours(batch_size * dims.vol())
    , rois(batch_size)
    , border_masks(batch_size * dims.vol())
    , division_ready(batch_size)
    , max_dist(batch_size)
    , protocols(batch_size)
    , rand_state(batch_size * CuBlockDimX * CuBlockDimY)
    , results(batch_size)
    , h_data(batch_size * dims.vol()) {
  cudaStreamCreate(&stream_);
  std::vector<uint32_t> h_seeds(batch_size * CuBlockDimX * CuBlockDimY);
  std::mt19937 rand{seed};
  std::generate(h_seeds.begin(), h_seeds.end(), rand);
  auto seeds = device::buffer<uint32_t>::fromHost(h_seeds.data(), h_seeds.size(), stream_);
  rand_state.init(seeds.data(), stream_);
  cudaMemcpyAsync(d_params.get(), &params, sizeof(Parameters), cudaMemcpyHostToDevice, stream_);
  populateLattices();
}

void Simulation::sendData(const HostGrid<Site> &grid, const Protocol &protocol, uint32_t multi) {
  assert(filled_samples + multi <= batch_size);
  assert(grid.view().dims == dims);
  for (uint32_t i = filled_samples; i < filled_samples + multi; ++i) {
    auto view = grid.view();
    data.copyHost(view.data, dims.vol(), dims.vol() * i, stream_);
    KERNEL_DEBUG("data")
    protocols.copyHost(&protocol, 1, i, stream_);
    KERNEL_DEBUG("protocol")
  }
  filled_samples += multi;
}

void Simulation::step() {
  if (step_ % 64 == 0) {
    findROIs(rois.data(), border_masks.data(), lattices.data(), batch_size, stream_);
  }
  diffuse();
  simulateCells();
  cellDivision();
  ++step_;
}

void Simulation::diffuse() {
  batchDiffusion(lattices.data(), rois.data(), border_masks.data(), params.diffusion_params,
                 params.external_levels, params.time_step/params.diffusion_params.time_step,
                 dims, batch_size, stream_);
}

void Simulation::simulateCells() {
  detail::cellSimulationKernel
    <<<batch_size, dim3(CuBlockDimX, CuBlockDimY), CuBlockDimX*CuBlockDimY, stream_>>>
    (lattices.data(), rois.data(), d_params.get(), rand_state.states(), division_ready.data(),
        protocols.data(), step_);
  KERNEL_DEBUG("simulate cells")
}

void Simulation::cellDivision() {
  detail::cellDivisionKernel<<<batch_size, dim3(CuBlockDimX/2, CuBlockDimY/2), 0, stream_>>>
    (lattices.data(), d_params.get(), division_ready.data(), rand_state.states());
  KERNEL_DEBUG("cell division")
}
void Simulation::updateROIs() {
  findTumorsBoundaries(lattices.data(), rois.data(), batch_size, stream_);
}

void Simulation::getResults(uint32_t *h_data) {
  countLiving(results.data(), data.data(), dims, batch_size);
  cudaMemcpyAsync(h_data, results.data(), batch_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
                  stream_);
  sync();
}
void Simulation::run(uint32_t nsteps) {
  assert(filled_samples == batch_size);
  for (uint32_t s = 0; s < nsteps; ++s) {
    step();
  }
}

void Simulation::getData(Site *h_data, uint32_t sample) {
  assert(sample < batch_size);
  cudaMemcpyAsync(h_data, data.data() + sample * dims.vol(),
           dims.vol() * sizeof(Site), cudaMemcpyDeviceToHost, stream_);
  sync();
}

void Simulation::sync() {
  cudaStreamSynchronize(stream_);
}

void Simulation::reset() {
  sync();
  step_ = 0;
  filled_samples = 0;
}

}  // namespace emt6ro
