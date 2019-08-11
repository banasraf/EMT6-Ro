#ifndef SRC_EMT6RO_DIVISION_CELL_DIVISION_H_
#define SRC_EMT6RO_DIVISION_CELL_DIVISION_H_

#include "emt6ro/cell/cell.h"

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

}  // namespace emt6ro

#endif  // SRC_EMT6RO_DIVISION_CELL_DIVISION_H_
