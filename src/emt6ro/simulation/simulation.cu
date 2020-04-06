#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include "emt6ro/common/debug.h"
#include "emt6ro/common/grid.h"
#include "emt6ro/diffusion/diffusion.h"
#include "emt6ro/division/cell-division.h"
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/statistics/statistics.h"

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
                                     Parameters params, curandState_t *rand_states,
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
  auto dose = protocol.getDose(step);
  GRID_FOR(start_r, start_c, roi.dims.height + start_r, roi.dims.width + start_c) {
    auto &site = grid(r, c);
    division_ready[tid] = division_ready[tid] ||
                          site.step(params, vacant_neighbours[subi], dose, rand);
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

}  // namespace detail

void Simulation::populateLattices() {
  auto blocks = (batch_size + 1023) / 1024;
  auto block_size = (batch_size > 1024) ? 1024 : batch_size;
  detail::populateGridViewsKernel<<<blocks, block_size, 0, str.stream_>>>
    (lattices.data(), batch_size, dims, data.data());
  KERNEL_DEBUG("populate")
}

Simulation::Simulation(uint32_t batch_size, const Parameters &parameters, uint32_t seed)
    : batch_size(batch_size)
    , dims({parameters.lattice_dims.height+2, parameters.lattice_dims.width+2})
    , params(parameters)
    , data(batch_size * dims.vol())
    , protocols(batch_size)
    , lattices(batch_size)
    , rois(batch_size)
    , border_masks(batch_size * dims.vol())
    , division_ready(batch_size)
    , rand_state(batch_size * CuBlockDimX * CuBlockDimY)
    , results(batch_size) {
  std::vector<uint32_t> h_seeds(batch_size * CuBlockDimX * CuBlockDimY);
  std::mt19937 rand{seed};
  std::generate(h_seeds.begin(), h_seeds.end(), rand);
  auto seeds = device::buffer<uint32_t>::fromHost(h_seeds.data(), h_seeds.size(), str.stream_);
  rand_state.init(seeds.data(), str.stream_);
  populateLattices();
}

void Simulation::sendData(const HostGrid<Site> &grid, const Protocol &protocol, uint32_t multi) {
  assert(filled_samples + multi <= batch_size);
  assert(grid.view().dims == dims);
  for (uint32_t i = filled_samples; i < filled_samples + multi; ++i) {
    auto view = grid.view();
    data.copyHost(view.data, dims.vol(), dims.vol() * i, str.stream_);
    KERNEL_DEBUG("data")
    protocols.copyHost(&protocol, 1, i, str.stream_);
    KERNEL_DEBUG("protocol")
  }
  filled_samples += multi;
}

void Simulation::step() {
  if (step_ % 64 == 0) {
    updateROIs();
  }
  diffuse();
  simulateCells();
  cellDivision();
  ++step_;
}

void Simulation::diffuse() {
  batchDiffusion(lattices.data(), rois.data(), border_masks.data(), params.diffusion_params,
                 params.external_levels, params.time_step/params.diffusion_params.time_step,
                 dims, batch_size, str.stream_);
}

void Simulation::simulateCells() {
  detail::cellSimulationKernel
    <<<batch_size, dim3(CuBlockDimX, CuBlockDimY), CuBlockDimX*CuBlockDimY, str.stream_>>>
    (lattices.data(), rois.data(), params, rand_state.states(), division_ready.data(),
        protocols.data(), step_);
  KERNEL_DEBUG("simulate cells")
}

void Simulation::cellDivision() {
  batchCellDivision(lattices.data(), params, division_ready.data(), rand_state.states(),
                    batch_size, str.stream_);
}
void Simulation::updateROIs() {
  findROIs(rois.data(), border_masks.data(), lattices.data(), batch_size, str.stream_);
}

void Simulation::getResults(uint32_t *h_results) {
  countLiving(results.data(), data.data(), dims, batch_size, str.stream_);
  cudaMemcpyAsync(h_results, results.data(), batch_size * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, str.stream_);
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
            dims.vol() * sizeof(Site), cudaMemcpyDeviceToHost, str.stream_);
  sync();
}

void Simulation::sync() {
  cudaStreamSynchronize(str.stream_);
}

}  // namespace emt6ro
