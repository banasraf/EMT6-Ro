#ifndef SRC_EMT6RO_CELL_CELL_H_
#define SRC_EMT6RO_CELL_CELL_H_

/// @file cell.h

#include <cuda_runtime.h>
#include <cstdint>
#include "emt6ro/common/substrates.h"
#include "emt6ro/parameters/parameters.h"

namespace emt6ro {

/// @brief Cells group representation.
struct Cell {
  /// @brief Metabolic mode of a cell.
  enum class MetabolicMode : uint8_t {
    AEROBIC_PROLIFERATION = 0,
    ANAEROBIC_PROLIFERATION = 1,
    AEROBIC_QUIESCENCE = 2,
    ANAEROBIC_QUIESCENCE = 3,
  };

  /// @brief Cell's proliferation cycle phase.
  enum class CyclePhase : uint8_t { G1 = 0, S = 1, G2 = 2, M = 3, D = 4 };

  /**
   * @brief Ending points of all cycle phases.
   * Every value represents the proliferation time
   * at which corresponding phase is finished.
   */
  struct CycleTimes {
    union {
      float times[5];
      struct {
        float g1;
        float s;
        float g2;
        float m;
        float d;
      };
    };
  };

  float time_in_repair;
  float irradiation;
  float repair_delay_time;
  float proliferation_time;
  CycleTimes cycle_times;
  MetabolicMode mode;
  CyclePhase phase;

  /**
   * Cell metabolism. Modifies `site_substrates`.
   * @param site_substrates substrate levels on a site occupied by the cell
   * @param metabolism parameters
   */
  __host__ __device__ void metabolise(Substrates &site_substrates,
                                      const Parameters::Metabolism &metabolism);

  /**
   * Progress proliferation clock and update cycle phase.
   * Returns true if major cycle change has occurred
   * (i.e. **G1** → **S**, **S** → **G2**, **G2** → **D**)
   * @param time_step simulation time step
   */
  __host__ __device__ bool progressClock(float time_step);
};

}  // namespace emt6ro

#endif  // SRC_EMT6RO_CELL_CELL_H_
