#ifndef EMT6RO_DIVISION_CELL_DIVISION_H_
#define EMT6RO_DIVISION_CELL_DIVISION_H_

#include <cmath>
#include "emt6ro/common/random-engine.h"
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
  auto new_cell = createCell(params, rand);
  new_cell.irradiation = cell.irradiation;
  if (cell.irradiation > 0)
    new_cell.calcDelayTime(params.cell_repair);
  return new_cell;
}

__host__ __device__ static inline Coords mapToDiagNeighbour(int32_t r, int32_t c, uint8_t num) {
  const int8_t vert = (num & 1U) * 2 - 1;
  const int8_t hor = ((num & 2U) >> 1U) * 2 - 1;
  return {r + vert, c + hor};
}

__host__ __device__ static inline Coords mapToOrthoNeighbour(int32_t r, int32_t c, uint8_t num) {
  const bool which = num & 1U;
  const int8_t diff = ((num & 2U) >> 1U) * 2 - 1;
  return which ? Coords{r + diff, c} : Coords{r, c + diff};
}

template <typename R>
__host__ __device__ Coords chooseNeighbour(uint32_t r, uint32_t c, R &rand) {
  static constexpr float diagProb = 4.;
  static constexpr float orthoProb = 4. * M_SQRT2;
  auto score = rand.uniform() * (diagProb + orthoProb);
  if (score < diagProb) {
    return mapToDiagNeighbour(r, c, static_cast<uint8_t>(score));
  } else {
    return mapToOrthoNeighbour(r, c, static_cast<uint8_t>((score - diagProb) / M_SQRT2));
  }
}

__device__ void divideCells(GridView<Site> &lattice, const Parameters &params, CuRandEngine &rand);

void batchCellDivision(GridView<Site> *lattices, Parameters params, const bool *division_ready,
                       curandState_t *rand_states, int32_t batch_size,
                       cudaStream_t stream = nullptr);

}  // namespace emt6ro

#endif  // EMT6RO_DIVISION_CELL_DIVISION_H_
