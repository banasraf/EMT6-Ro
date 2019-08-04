#include "emt6ro/cell/cell.h"

namespace emt6ro {

namespace {

__host__ __device__ Cell::CyclePhase progressPhase(Cell::CyclePhase current, bool change) {
  return static_cast<Cell::CyclePhase>(static_cast<uint8_t>(current) +
                                       static_cast<uint8_t>(change));
}

}  // namespace

__host__ __device__ void Cell::metabolise(Substrates &site_substrates,
                                          const Parameters::Metabolism &metabolism) {
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
__host__ __device__ bool Cell::tryProliferating(const Substrates &levels,
                                                const Parameters &params) {
  if (levels.cho >= params.metabolism.aerobic_proliferation.cho &&
      levels.ox >= params.metabolism.aerobic_proliferation.ox &&
      levels.gi < params.quiescence_gi) {
    mode = MetabolicMode::AEROBIC_PROLIFERATION;
    return true;
  } else if (levels.cho >= params.metabolism.anaerobic_proliferation.cho &&
             levels.ox >= params.metabolism.anaerobic_proliferation.ox &&
             levels.gi < params.quiescence_gi) {
    mode = MetabolicMode::ANAEROBIC_PROLIFERATION;
    return true;
  }
  return false;
}

__host__ __device__ bool Cell::tryQuiescence(const Substrates &levels, const Parameters &params) {
  if (phase != CyclePhase::G1 && phase != CyclePhase::G2) {
    return false;
  }
  if (levels.cho >= params.metabolism.aerobic_quiescence.cho &&
      levels.ox >= params.metabolism.aerobic_quiescence.ox &&
      levels.gi < params.death_gi) {
    mode = MetabolicMode::AEROBIC_QUIESCENCE;
    return true;
  } else if (levels.cho >= params.metabolism.anaerobic_quiescence.cho &&
      levels.ox >= params.metabolism.anaerobic_quiescence.ox &&
      levels.gi < params.death_gi) {
    mode = MetabolicMode::ANAEROBIC_QUIESCENCE;
    return true;
  }
  return false;
}

__host__ __device__ bool Cell::enterG1SStopping(float time_step, uint8_t vacant_neighbours) {
  return proliferation_time > (cycle_times.g1 - 2*time_step) &&
         phase == CyclePhase::G1 && vacant_neighbours == 0;
}

__host__ __device__ bool Cell::updateState(const Substrates &levels, const Parameters &params,
                                           uint8_t vacant_neighbors) {
  if (!enterG1SStopping(params.time_step, vacant_neighbors)) {
    if (tryProliferating(levels, params)) {
      return true;
    }
  }
  return tryQuiescence(levels, params);
}

}  // namespace emt6ro
