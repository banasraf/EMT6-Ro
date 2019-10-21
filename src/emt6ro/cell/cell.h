#ifndef SRC_EMT6RO_CELL_CELL_H_
#define SRC_EMT6RO_CELL_CELL_H_

/// @file cell.h

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
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

  /**
   * Set metabolic mode of a cell or decide it is dead due to environment conditions.
   * @param levels - substrate levels on a site occupied by the cell
   * @param params - simulation parameters
   * @param vacant_neighbours - number of vacant neighbouring sites
   * @return true if the cell is still alive, false otherwise
   */
  __host__ __device__ bool updateState(const Substrates &levels, const Parameters &params,
                                       uint8_t vacant_neighbours);

  __host__ __device__ void calcDelayTime(const Parameters::CellRepair &params);

  __host__ __device__ void irradiate(float dose, const Parameters::CellRepair &params);

  template <typename R>
  __host__ __device__ bool tryRepair(const Parameters::CellRepair &params, bool cycle_changed,
                                     float time_step, R &rand) {
    using std::exp;
    if (time_in_repair > 0 || (irradiation > 0 && cycle_changed)) {
      time_in_repair += time_step / 3600.f;
      if (time_in_repair >= repair_delay_time) {
        float death_prob =
            1 - params.survival_prob.coeff * exp(params.survival_prob.exp_coeff * irradiation);
        if (rand.uniform() < death_prob) {
          return false;
        } else {
          irradiation = 0.f;
          time_in_repair = 0.f;
          repair_delay_time = 0.f;
        }
      }
    }
    return true;
  }

 private:
  __host__ __device__ bool enterG1SStopping(float time_step, uint8_t vacant_neighbours);

  __host__ __device__ bool tryProliferating(const Substrates &levels, const Parameters &params);

  __host__ __device__ bool tryQuiescence(const Substrates &levels, const Parameters &params);
};

}  // namespace emt6ro

#endif  // SRC_EMT6RO_CELL_CELL_H_
