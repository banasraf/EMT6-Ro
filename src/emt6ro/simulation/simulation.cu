#include <algorithm>
#include <cassert>
#include <random>
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/diffusion/grid-diffusion.h"
#include "emt6ro/division/cell-division.h"

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
}

void copyLattice(device::buffer<Site> &data, uint32_t batch, const GridView<Site> &h_lattice) {
  assert(data.size() >= batch * h_lattice.dims.vol());
  for (uint32_t i = 0; i < batch; ++i) {
    data.copyHost(h_lattice.data, h_lattice.dims.vol(), i * h_lattice.dims.vol());
  }
}

__global__ void vacantNeighboursKernel(GridView<Site> *views, uint8_t *results) {
  auto &view = views[blockIdx.x];
  GridView<uint8_t> result_view{results + view.dims.vol() * blockIdx.x, view.dims};
  GRID_FOR(1, 1, view.dims.height - 1, view.dims.width - 1) {
    result_view(r, c) = view(r - 1, c - 1).isOccupied() +
                        view(r - 1, c).isOccupied() +
                        view(r - 1, c + 1).isOccupied() +
                        view(r, c - 1).isOccupied() +
                        view(r, c + 1).isOccupied() +
                        view(r + 1, c - 1).isOccupied() +
                        view(r + 1, c).isOccupied() +
                        view(r + 1, c + 1).isOccupied();
  }
}

__global__ void cellSimulationKernel(Site *sites, uint32_t sites_count, uint8_t *vacant,
                                     Parameters *params) {
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= sites_count) return;
  auto &site = sites[idx];
  if (site.state != Site::State::OCCUPIED) return;
  uint8_t alive = site.cell.updateState(sites[idx].substrates, *params, vacant[idx]);
  site.state = static_cast<Site::State>(alive);
  if (alive) {
    site.cell.metabolise(site.substrates, params->metabolism);
    site.cell.progressClock(params->time_step);
  }
}

__global__ void cellDivisionKernel(GridView<Site> *lattices, Parameters *params,
                                   curandState_t *rand_states) {
  curandState_t *rand_state =
      rand_states + blockDim.x * blockDim.y * blockIdx.x + blockDim.y * threadIdx.y + threadIdx.x;
  CuRandEngine rand(rand_state);
  auto &lattice = lattices[blockIdx.x];
  divideCells(lattice, *params, rand);
}

}  // namespace detail

Simulation Simulation::FromSingleHost(const GridView<Site>& lattice, uint32_t batch,
                                      const Parameters& params, uint32_t seed) {
  std::vector<uint32_t> h_seeds(batch * CuBlockDimX * CuBlockDimY);
  std::mt19937 rand{seed};
  std::generate(h_seeds.begin(), h_seeds.end(), rand);
  auto d_seeds = device::buffer<uint32_t>::fromHost(h_seeds.data(), h_seeds.size());
  Simulation simulation(lattice.dims, batch, params, d_seeds);
  detail::populateGridViews(simulation.lattices.data(), batch,
                            lattice.dims, simulation.data.data());
  detail::copyLattice(simulation.data, batch, lattice);
  return simulation;
}

void Simulation::step() {
  diffuse();
  calculateVacantNeighbours();
  simulateCells();
  cellDivision();
}

void Simulation::diffuse() {
  copySubstrates(diffusion_tmp_data.cho.data(),
                 diffusion_tmp_data.ox.data(),
                 diffusion_tmp_data.gi.data(),
                 data.data(), dims, batch_size);
  batchDiffuse(diffusion_tmp_data.cho.data(), dims, batch_size,
               params.diffusion_params.coeffs.cho, params.diffusion_params.time_step,
               static_cast<uint32_t>(params.time_step / params.diffusion_params.time_step));
  batchDiffuse(diffusion_tmp_data.ox.data(), dims, batch_size,
               params.diffusion_params.coeffs.ox, params.diffusion_params.time_step,
               static_cast<uint32_t>(params.time_step / params.diffusion_params.time_step));
  batchDiffuse(diffusion_tmp_data.gi.data(), dims, batch_size,
               params.diffusion_params.coeffs.gi, params.diffusion_params.time_step,
               static_cast<uint32_t>(params.time_step / params.diffusion_params.time_step));
  copySubstratesBack(data.data(),
                     diffusion_tmp_data.cho.data(),
                     diffusion_tmp_data.ox.data(),
                     diffusion_tmp_data.gi.data(),
                     dims, batch_size);
}

void Simulation::calculateVacantNeighbours() {
  detail::vacantNeighboursKernel<<<batch_size, dim3(CuBlockDimX, CuBlockDimY)>>>
    (lattices.data(), vacant_neighbours.data());
}

void Simulation::simulateCells() {
  auto sites_count = dims.vol() * batch_size;
  auto block_size = (sites_count < 1024) ? sites_count : 1024;
  auto blocks = (sites_count + 1023) / 1024;
  detail::cellSimulationKernel<<<blocks, block_size>>>
    (data.data(), sites_count, vacant_neighbours.data(), d_params.get());
}

void Simulation::cellDivision() {
  detail::cellDivisionKernel<<<batch_size, dim3(CuBlockDimX, CuBlockDimY)>>>
    (lattices.data(), d_params.get(), rand_state.states());
}

}  // namespace emt6ro