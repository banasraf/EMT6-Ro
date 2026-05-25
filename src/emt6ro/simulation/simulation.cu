#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "emt6ro/common/debug.h"
#include "emt6ro/common/grid.h"
#include "emt6ro/diffusion/diffusion.h"
#include "emt6ro/simulation/cell-division.h"
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/statistics/statistics.h"
#include "emt6ro/common/cuda-utils.h"
#include "emt6ro/common/stack.cuh"
#include "emt6ro/common/error.h"

#ifdef EMT6RO_NVTX
#include <nvToolsExt.h>
#define EMT6RO_NVTX_RANGE(name) ::nvtxRangePushA(name)
#define EMT6RO_NVTX_END() ::nvtxRangePop()
#else
#define EMT6RO_NVTX_RANGE(name) ((void)0)
#define EMT6RO_NVTX_END() ((void)0)
#endif

#ifdef EMT6RO_TIMING
// Record the start event for kernel slot `idx` on the simulation stream.
#define EMT6RO_TIME_BEGIN(idx) ::cudaEventRecord(evt_start_[idx], str.stream_)
// Record the stop event, wait for it, and add elapsed ms to `accumulator`.
// Per-step host-side synchronisation cost is the price of a per-kernel
// breakdown — turn EMT6RO_TIMING off for production runs.
#define EMT6RO_TIME_END(idx, accumulator) do {                                  \
    ::cudaEventRecord(evt_stop_[idx], str.stream_);                             \
    ::cudaEventSynchronize(evt_stop_[idx]);                                     \
    float _ms = 0.f;                                                            \
    ::cudaEventElapsedTime(&_ms, evt_start_[idx], evt_stop_[idx]);              \
    (accumulator) += _ms;                                                       \
  } while (0)
#else
#define EMT6RO_TIME_BEGIN(idx) ((void)0)
#define EMT6RO_TIME_END(idx, accumulator) ((void)0)
#endif

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

__host__ __device__ uint8_t vacantNeighbours(const GridView<Site> &grid, int16_t r, int16_t c) {
  return grid(r - 1, c - 1).isVacant() +
         grid(r - 1, c + 1).isVacant() +
         grid(r + 1, c - 1).isVacant() +
         grid(r + 1, c + 1).isVacant() +
         grid(r, c - 1).isVacant() +
         grid(r, c + 1).isVacant() +
         grid(r - 1, c).isVacant() +
         grid(r + 1, c).isVacant();
}

static const int kFindOccupiedNthreads = 32;
__global__ void findOccupied(GridView<Site> *lattices, uint32_t *occupied_b) {
  extern __shared__ uint32_t shmem[];
  auto lattice = lattices[blockIdx.x];
  uint32_t n = 0;
  Coords collection[1024 / kFindOccupiedNthreads];
  GRID_FOR(0, 0, lattice.dims.height - 1, lattice.dims.width - 1) {
    if (lattice(r, c).isOccupied()) {
      collection[n++] = Coords{r, c};
    }
  }
  shmem[threadIdx.x] = n;
  __syncthreads();
  uint32_t acc = 0;
  for (int i = 0; i < threadIdx.x; ++i) {
    acc += shmem[i];
  }
  uint32_t &n_occupied = occupied_b[blockIdx.x * 1024];
  auto *occupied = reinterpret_cast<Coords*>(&n_occupied + 1);
  for (int i = 0; i < n; ++i)
    occupied[acc + i] = collection[i];
  if (threadIdx.x == blockDim.x - 1)
    n_occupied = acc + n;
}

__global__ void cellSimulationKernel(GridView<Site> *grids, uint32_t *occupied_b,
                                     Parameters params, curandState_t *rand_states,
                                     Protocol *protocols, uint32_t step) {
  extern __shared__ uint64_t shm[];
  StackView<Coords> occupied(&occupied_b[blockIdx.x * 1024]);
  uint64_t division = 0;
  uint8_t vacant_neighbours[4];
  auto &grid = grids[blockIdx.x];
  const auto &protocol = protocols[blockIdx.x];
  curandState_t *rand_state =
      rand_states + blockDim.x * blockIdx.x + threadIdx.x;
  CuRandEngine rand(rand_state);
  uint8_t subi = 0;
  for (auto coords : dev_iter(occupied)) {
    vacant_neighbours[subi] = vacantNeighbours(grid, coords.r, coords.c);
    ++subi;
  }
  __syncthreads();
  subi = 0;
  auto dose = protocol.getDose(step);
  for (auto coords : dev_iter(occupied)) {
    auto &site = grid(coords);
    auto d = site.step(params, vacant_neighbours[subi], dose, rand);
    if (d) {
      auto child_coords = chooseNeighbour(coords.r, coords.c, rand);
      if (grid(child_coords).isVacant()) {
        division = Coords2(coords, child_coords).encode();
      }
    }
    ++subi;
  }
  division = block_reduce(division, shm,
                          [](uint64_t a, uint64_t b){return a | (b * (uint64_t)(!a));});
  if (threadIdx.x == 0 && division) {
    Coords2 coords = Coords2::decode(division);
    auto parent = coords[0];
    auto child = coords[1];
    grid(child).state = Site::State::OCCUPIED;
    grid(child).cell = divideCell(grid(parent).cell, params, rand);
    occupied.push(child);
#ifdef EMT6RO_INSTRUMENT
    atomicAdd(&g_instr_divisions, 1ULL);
#endif
  }
}

}  // namespace detail

void Simulation::populateLattices() {
  auto mbs = CuBlockDimX * CuBlockDimY;
  auto blocks = div_ceil(batch_size, mbs);
  auto block_size = (batch_size > mbs) ? mbs : batch_size;
  detail::populateGridViewsKernel<<<blocks, block_size, 0, str.stream_>>>
    (lattices.data(), batch_size, dims, data.data());
  KERNEL_DEBUG("populate")
}

Simulation::Simulation(uint32_t batch_size, const Parameters &parameters, uint32_t seed)
    : batch_size(batch_size)
    , dims(Dims(parameters.lattice_dims.height+2, parameters.lattice_dims.width+2))
    , params(parameters)
    , data(batch_size * dims.vol())
    , protocols(batch_size)
    , lattices(batch_size)
    , rois(batch_size)
    , border_masks(batch_size * dims.vol())
    , occupied(batch_size * 1024)
    , rand_state(batch_size * simulate_num_threads)
    , results(batch_size) {
  rand_state.init(seed, str.stream_);
  populateLattices();
#ifdef EMT6RO_TIMING
  for (int i = 0; i < 5; ++i) {
    cudaEventCreate(&evt_start_[i]);
    cudaEventCreate(&evt_stop_[i]);
  }
  events_inited_ = true;
#endif
}

#ifdef EMT6RO_TIMING
Simulation::~Simulation() {
  if (events_inited_) {
    for (int i = 0; i < 5; ++i) {
      cudaEventDestroy(evt_start_[i]);
      cudaEventDestroy(evt_stop_[i]);
    }
    events_inited_ = false;
  }
}
#endif

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
  filled_protocols += multi;
}

void Simulation::step() {
  EMT6RO_NVTX_RANGE("step");
  if (step_ % 128 == 0) {
    EMT6RO_NVTX_RANGE("findOccupied");
    EMT6RO_TIME_BEGIN(0);
    detail::findOccupied
    <<<batch_size, detail::kFindOccupiedNthreads,
       detail::kFindOccupiedNthreads * sizeof(uint32_t), str.stream_>>>
    (lattices.data(), occupied.data());
    EMT6RO_TIME_END(0, timers_.findOccupied_ms);
    EMT6RO_NVTX_END();
  }
  if (step_ % 32 == 0) {
    EMT6RO_NVTX_RANGE("updateROIs");
    EMT6RO_TIME_BEGIN(1);
    updateROIs();
    EMT6RO_TIME_END(1, timers_.updateROIs_ms);
    EMT6RO_NVTX_END();
  }
  {
    EMT6RO_NVTX_RANGE("diffuse");
    EMT6RO_TIME_BEGIN(2);
    diffuse();
    EMT6RO_TIME_END(2, timers_.diffuse_ms);
    EMT6RO_NVTX_END();
  }
  {
    EMT6RO_NVTX_RANGE("simulateCells");
    EMT6RO_TIME_BEGIN(3);
    simulateCells();
    EMT6RO_TIME_END(3, timers_.simulateCells_ms);
    EMT6RO_NVTX_END();
  }
#ifdef EMT6RO_TIMING
  ++timers_.n_steps;
#endif
  ++step_;
  EMT6RO_NVTX_END();
}

void Simulation::diffuse() {
  batchDiffusion(lattices.data(), rois.data(), border_masks.data(), params.diffusion_params,
                 params.external_levels, params.time_step/params.diffusion_params.time_step,
                 dims, batch_size, str.stream_);
}

void Simulation::simulateCells() {
  detail::cellSimulationKernel
    <<<batch_size, simulate_num_threads, sizeof(uint64_t)*simulate_num_threads/32, str.stream_>>>
    (lattices.data(), occupied.data(), params, rand_state.states(),
     protocols.data(), step_);
  KERNEL_DEBUG("simulate cells")
}

void Simulation::updateROIs() {
  findROIs(rois.data(), border_masks.data(), lattices.data(), batch_size, str.stream_);
}

void Simulation::getResults(uint32_t *h_results) {
  EMT6RO_NVTX_RANGE("countLiving");
  EMT6RO_TIME_BEGIN(4);
  countLiving(results.data(), data.data(), dims, batch_size, str.stream_);
  cudaMemcpyAsync(h_results, results.data(), batch_size * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, str.stream_);
  EMT6RO_TIME_END(4, timers_.countLiving_ms);
  sync();
  EMT6RO_NVTX_END();
}

void Simulation::run(uint32_t nsteps) {
  ENFORCE(filled_samples == batch_size, "");
  for (uint32_t s = 0; s < nsteps; ++s) {
    step();
  }
}

void Simulation::getData(Site *h_data, uint32_t sample) {
  ENFORCE(sample < batch_size, make_string("Cannot read data from sample ",
                                           sample, ". Batch size: ", batch_size));
  cudaMemcpyAsync(h_data, data.data() + sample * dims.vol(),
            dims.vol() * sizeof(Site), cudaMemcpyDeviceToHost, str.stream_);
  sync();
}

void Simulation::sync() {
  cudaStreamSynchronize(str.stream_);
}

void Simulation::reset() {
  sync();
  step_ = 0;
  filled_samples = 0;
}

void Simulation::setState(const Site *state) {
  cudaMemcpyAsync(data.data(), state, batch_size * dims.vol() * sizeof(Site), 
                  cudaMemcpyDeviceToDevice, str.stream_);
  KERNEL_DEBUG("copy data");
  filled_samples = batch_size;
}

void Simulation::setProtocols(const Protocol *ps) {
  cudaMemcpyAsync(protocols.data(), ps, batch_size * sizeof(Protocol), 
                  cudaMemcpyHostToDevice, str.stream_);
  KERNEL_DEBUG("copy protocols");
  filled_protocols = batch_size;
}

}  // namespace emt6ro
