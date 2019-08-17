#include "cell-division.h"


namespace emt6ro {

__device__ void divideCells(GridView<Site> &lattice, const Parameters &params, CuRandEngine &rand) {
  for (uint32_t conv_r = 0; conv_r < 3; ++conv_r) {
    for (uint32_t conv_c = 0; conv_c < 3; ++conv_c) {
      GRID_FOR(0, 0, (lattice.dims.height + 2) / 3, (lattice.dims.width + 2) / 3) {
          const auto rr = 3 * r + conv_r;
          const auto cc = 3 * c + conv_c;
          if (rr < lattice.dims.height - 1 && cc < lattice.dims.width) {
            if (lattice(rr, cc).state == Site::State::OCCUPIED) {
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

}  // namespace emt6ro