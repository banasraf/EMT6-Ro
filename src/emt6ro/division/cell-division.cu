#include "emt6ro/division/cell-division.h"
#include "emt6ro/common/debug.h"

namespace emt6ro {

__device__ void divideCells(GridView<Site> &lattice, const Parameters &params, CuRandEngine &rand) {
  for (uint32_t conv_r = 0; conv_r < 3; ++conv_r) {
    for (uint32_t conv_c = 0; conv_c < 3; ++conv_c) {
      GRID_FOR(0, 0, (lattice.dims.height + 2) / 3, (lattice.dims.width + 2) / 3) {
          const auto rr = 3 * r + conv_r;
          const auto cc = 3 * c + conv_c;
          if (rr < lattice.dims.height - 1 && cc < lattice.dims.width - 1) {
            if (lattice(rr, cc).isOccupied()) {
              auto &cell = lattice(rr, cc).cell;
              if (cell.phase == Cell::CyclePhase::D) {
                auto neighbour = chooseNeighbour(rr, cc, rand);
                if (!lattice(neighbour).isOccupied()) {
                  lattice(neighbour).cell = divideCell(cell, params, rand);
                  lattice(neighbour).state = Site::State::OCCUPIED;
                }
              }
            }
          }
        }
      __syncthreads();
    }
  }
}

__global__ void cellDivisionKernel(GridView<Site> *lattices, Parameters params,
                                   const bool *division_ready, curandState_t *rand_states) {
  if (!division_ready[blockIdx.x]) return;
  curandState_t *rand_state =
      rand_states + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * threadIdx.y + threadIdx.x;
  CuRandEngine rand(rand_state);
  auto &lattice = lattices[blockIdx.x];
  divideCells(lattice, params, rand);
}

void batchCellDivision(GridView<Site> *lattices, Parameters params, const bool *division_ready,
                       curandState_t *rand_states, int32_t batch_size,
                       cudaStream_t stream) {
  cellDivisionKernel<<<batch_size, dim3(CuBlockDimX/2, CuBlockDimY/2), 0, stream>>>
  (lattices, params, division_ready, rand_states);
  KERNEL_DEBUG("cell division")
}


}  // namespace emt6ro
