#include "emt6ro/cell/cell.h"

namespace emt6ro {

namespace {

__host__ __device__ Cell::CyclePhase progressPhase(Cell::CyclePhase current, bool change) {
  return static_cast<Cell::CyclePhase>(
      static_cast<uint8_t>(current) + static_cast<uint8_t>(change));
}

}  // namespace

__host__ __device__ void Cell::metabolise(Substrates& site_substrates,
                                          const Parameters::Metabolism& metabolism) {
  site_substrates -= metabolism.values[static_cast<uint8_t>(mode)];
}

__host__ __device__ bool Cell::progressClock(float time_step) {
  if (mode == MetabolicMode::AEROBIC_PROLIFERATION ||
      mode == MetabolicMode::ANAEROBIC_PROLIFERATION) {
    proliferation_time += time_step / 3600;
    CyclePhase current = phase;
    phase = progressPhase(current,
                          proliferation_time > cycle_times.times[static_cast<uint8_t>(current)]);
    return phase != current && phase != CyclePhase::D;
  }
  return false;
}
__host__ __device__ bool Cell::updateState(const Substrates& levels,
                                           const Parameters& params,
                                           uint8_t vacant_neighbors){

}

}  // namespace emt6ro
