#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include "emt6ro/diffusion/grid-diffusion.h"
#include "emt6ro/division/cell-division.h"
#include "emt6ro/simulation/simulation.h"

namespace emt6ro {

namespace detail {

__global__ void populateGridViewsKernel(GridView<Site> *views, uint32_t size,
                                        Dims dims, Site *origin) {
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < size) {
    views[idx].data = origin + dims.vol() * idx;
    views[idx].dims = dims;
  }
}

void populateGridViews(GridView<Site> *views, uint32_t size,
                       Dims dims, Site *origin) {
  auto blocks = (size + 1023) / 1024;
  auto block_size = (size > 1024) ? 1024 : size;
  populateGridViewsKernel<<<blocks, block_size>>>(views, size, dims, origin);
  KERNEL_DEBUG("populate")
}

void copyLattice(device::buffer<Site> &data, uint32_t batch, const GridView<Site> &h_lattice) {
  assert(data.size() >= batch * h_lattice.dims.vol());
  for (uint32_t i = 0; i < batch; ++i) {
    data.copyHost(h_lattice.data, h_lattice.dims.vol(), i * h_lattice.dims.vol());
  }
}

void copyProtocols(device::buffer<Protocol> &protocols, const Protocol &protocol, uint32_t batch) {
  for (uint32_t i = 0; i < batch; ++i) {
    protocols.copyHost(&protocol, 1, i);
    KERNEL_DEBUG("copy protocols 2")
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
      uint8_t alive = site.cell.updateState(site.substrates, *params, vn, site.meta);
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

Simulation Simulation::FromSingleHost(const GridView<Site>& lattice, uint32_t batch,
                                      const Parameters& params, const Protocol &protocol,
                                      uint32_t seed) {
  std::vector<uint32_t> h_seeds(batch * CuBlockDimX * CuBlockDimY);
  std::mt19937 rand{seed};
  std::generate(h_seeds.begin(), h_seeds.end(), rand);
  auto d_seeds = device::buffer<uint32_t>::fromHost(h_seeds.data(), h_seeds.size());
  Simulation simulation(lattice.dims, batch, params, d_seeds);
  detail::populateGridViews(simulation.lattices.data(), batch,
                            lattice.dims, simulation.data.data());
  detail::copyProtocols(simulation.protocols, protocol, batch);
  detail::copyLattice(simulation.data, batch, lattice);
  return simulation;
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
  batchDiffuse2(lattices.data(), rois.data(), diffusion_tmp_data.data(), Dims{51, 51},
                params.diffusion_params.coeffs, params.external_levels, batch_size,
                params.diffusion_params.time_step, 24);
}

void Simulation::simulateCells() {
  detail::cellSimulationKernel<<<batch_size, dim3(32, 32), 51*51* sizeof(uint8_t)>>>
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

}  // namespace emt6ro