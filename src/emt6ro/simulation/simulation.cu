#include <emt6ro/common/grid.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include "emt6ro/common/debug.h"
#include "emt6ro/diffusion/grid-diffusion.h"
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
                                     Parameters *params, curandState_t *rand_states,
                                     Protocol *protocols, uint32_t step) {
  extern __shared__ uint8_t vacant_neighbours_mem[];
  const auto roi = rois[blockIdx.x];
  auto &grid = grids[blockIdx.x];
  const auto &protocol = protocols[blockIdx.x];
  GridView<uint8_t> vacant_neighbours{vacant_neighbours_mem, roi.dims};
  const auto start_r = roi.origin.r;
  const auto start_c = roi.origin.c;
  curandState_t *rand_state =
      rand_states + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x;
  CuRandEngine rand(rand_state);
  GRID_FOR(start_r, start_c, roi.dims.height + start_r, roi.dims.width + start_c) {
    vacant_neighbours(r - start_r, c - start_c) = vacantNeighbours(grid, r, c);
  }
  __syncthreads();
  GRID_FOR(start_r, start_c, roi.dims.height + start_r, roi.dims.width + start_c) {
    auto &site = grid(r, c);
    if (site.isOccupied()) {
      const auto vn = vacant_neighbours(r - start_r, c - start_c);
      uint8_t alive = site.cell.updateState(site.substrates, *params, vn);
      if (alive) {
        auto dose = protocol.getDose(step);
        if (dose) site.cell.irradiate(dose, params->cell_repair);
        site.cell.metabolise(site.substrates, params->metabolism);
        bool cycle_changed = site.cell.progressClock(params->time_step);
        alive = site.cell.tryRepair(params->cell_repair, cycle_changed, params->time_step, rand);
        if (!alive) site.state = Site::State::VACANT;
      } else {
        site.state = Site::State::VACANT;
      }
    }
  }
}

__global__ void cellDivisionKernel(GridView<Site> *lattices, Parameters *params,
                                   curandState_t *rand_states) {
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
  detail::populateGridViewsKernel<<<blocks, block_size>>>
    (lattices.data(), batch_size, dims, data.data());
  KERNEL_DEBUG("populate")
}

Simulation::Simulation(Dims dims, uint32_t batch_size, const Parameters &parameters, uint32_t seed)
    : batch_size(batch_size)
    , dims(dims)
    , params(parameters)
    , d_params(device::alloc_unique<Parameters>(1))
    , data(batch_size * dims.vol())
    , filled_samples(0)
    , lattices(batch_size)
    , diffusion_tmp_data(batch_size * dims.vol())
    , vacant_neighbours(batch_size * dims.vol())
    , rois(batch_size)
    , protocols(batch_size)
    , rand_state(batch_size * CuBlockDimX * CuBlockDimY)
    , results(batch_size) {
  std::vector<uint32_t> h_seeds(batch_size * CuBlockDimX * CuBlockDimY);
  std::mt19937 rand{seed};
  std::generate(h_seeds.begin(), h_seeds.end(), rand);
  auto seeds = device::buffer<uint32_t>::fromHost(h_seeds.data(), h_seeds.size());
  rand_state.init(seeds.data());
  cudaMemcpy(d_params.get(), &params, sizeof(Parameters), cudaMemcpyHostToDevice);
  populateLattices();
}

void Simulation::sendData(const HostGrid<Site> &grid, const Protocol &protocol, uint32_t multi) {
  assert(filled_samples + multi <= batch_size);
  assert(grid.view().dims == dims);
  for (uint32_t i = filled_samples; i < filled_samples + multi; ++i) {
    auto view = grid.view();
    data.copyHost(view.data, dims.vol(), dims.vol() * i);
    KERNEL_DEBUG("data")
    protocols.copyHost(&protocol, 1, i);
    KERNEL_DEBUG("protocol")
  }
  filled_samples += multi;
}

void Simulation::step() {
  if (step_ % 128 == 0)
    updateROIs();
  diffuse();
  simulateCells();
  cellDivision();
  ++step_;
}

void Simulation::diffuse() {
  batchDiffuse(lattices.data(), rois.data(), diffusion_tmp_data.data(),
               Dims{dims.height - 2, dims.width - 2},
               params.diffusion_params.coeffs, params.external_levels, batch_size,
               params.diffusion_params.time_step,
               static_cast<int32_t>(params.time_step / params.diffusion_params.time_step));
}

void Simulation::simulateCells() {
  auto mem_size = (dims.height - 2) * (dims.width - 2) * sizeof(uint8_t);
  detail::cellSimulationKernel
    <<<batch_size, dim3(CuBlockDimX, CuBlockDimY), mem_size>>>
    (lattices.data(), rois.data(), d_params.get(), rand_state.states(), protocols.data(), step_);
  KERNEL_DEBUG("simulate cells")
}

void Simulation::cellDivision() {
  detail::cellDivisionKernel<<<batch_size, dim3(CuBlockDimX/2, CuBlockDimY/2)>>>
    (lattices.data(), d_params.get(), rand_state.states());
  KERNEL_DEBUG("cell division")
}
void Simulation::updateROIs() {
  findTumorsBoundaries(lattices.data(), rois.data(), batch_size);
}

void Simulation::getResults(uint32_t *h_data) {
  countLiving(results.data(), data.data(), dims, batch_size);
  cudaMemcpy(h_data, results.data(), batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
void Simulation::run(uint32_t nsteps) {
  for (uint32_t s = 0; s < nsteps; ++s) {
    step();
  }
}

void Simulation::getData(Site *h_data, uint32_t sample) {
  assert(sample < batch_size);
  cudaMemcpy(h_data, data.data() + sample * dims.vol(),
      dims.vol() * sizeof(Site), cudaMemcpyDeviceToHost);
}

}  // namespace emt6ro
