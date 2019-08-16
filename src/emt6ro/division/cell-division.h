#ifndef SRC_EMT6RO_DIVISION_CELL_DIVISION_H_
#define SRC_EMT6RO_DIVISION_CELL_DIVISION_H_

#include <cmath>
#include "emt6ro/site/site.h"

namespace emt6ro {

template <typename R>
__host__ __device__ Cell createCell(const Parameters &params, R &rand) {
  Cell cell{};
  cell.cycle_times.times[0] = rand.normal(params.cycle_times.params[0]);
  cell.cycle_times.times[1] = cell.cycle_times.times[0] + rand.normal(params.cycle_times.params[1]);
  cell.cycle_times.times[2] = cell.cycle_times.times[1] + rand.normal(params.cycle_times.params[2]);
  cell.cycle_times.times[3] = cell.cycle_times.times[2] + rand.normal(params.cycle_times.params[3]);
  cell.cycle_times.times[4] = cell.cycle_times.times[3] + rand.normal(params.cycle_times.params[4]);
  return cell;
}

template <typename R>
__host__ __device__ Cell divideCell(Cell &cell, const Parameters &params, R &rand) {
  cell.phase = Cell::CyclePhase::G1;
  cell.proliferation_time = 0;
  return createCell(params, rand);
}

__host__ __device__ Coords mapToDiagNeighbour(uint32_t r, uint32_t c, uint8_t num) {
  const int8_t vert = (num & 1U) * 2 - 1;
  const int8_t hor = ((num & 2U) >> 1U) * 2 - 1;
  return {r + vert, c + hor};
}

__host__ __device__ Coords mapToOrthoNeighbour(uint32_t r, uint32_t c, uint32_t num) {
  const bool which = num & 1U;
  const int8_t diff = ((num & 2U) >> 1U) * 2 - 1;
  return which ? Coords{r + diff, c} : Coords{r, c + diff};
}

template <typename R>
__host__ __device__ Coords chooseNeighbour(uint32_t r, uint32_t c, R &rand) {
  static constexpr float diagProb = 4.;
  static constexpr float orthoProb = 4. * M_SQRT2f32;
  auto score = rand.uniform() * (diagProb + orthoProb);
  if (score < diagProb) {
    return mapToDiagNeighbour(r, c, static_cast<uint8_t>(score));
  } else {
    return mapToOrthoNeighbour(r, c, static_cast<uint8_t>((score - diagProb) / M_SQRT2f32));
  }
}

template <typename R>
__device__ void divideCells(GridView<Site> &lattice, const Parameters &params, R &rand) {
  for (uint32_t conv_r = 0; conv_r < 3; ++conv_r) {
    for (uint32_t conv_c = 0; conv_c < 3; ++conv_c) {
      GRID_FOR(0, 0, (lattice.dims.height + 2) / 3, (lattice.dims.width + 2) / 3) {
        const auto rr = 3 * r + conv_r;
        const auto cc = 3 * c + conv_c;
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
      __syncthreads();
    }
  }
}

}  // namespace emt6ro

#endif  // SRC_EMT6RO_DIVISION_CELL_DIVISION_H_
