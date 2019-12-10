#ifndef EMT6RO_SITE_SITE_H_
#define EMT6RO_SITE_SITE_H_

#include <cstdint>
#include "emt6ro/cell/cell.h"
#include "emt6ro/parameters/parameters.h"

namespace emt6ro {

/// @brief Representation of a site on a simulation lattice
struct Site {
  enum class State : uint8_t {
    VACANT = 0,  //!< no living cancerous cells on a site
    OCCUPIED = 1,  //!< site occupied by living cancerous cells
    MOCKED = 2  //!<  dummy state for border sites
  };

  Substrates substrates;  //!< site's substrates levels
  Cell cell;
  State state;

  __host__ __device__ inline uint8_t isOccupied() const {
    return state == State::OCCUPIED;
  }

  __host__ __device__ inline uint8_t isVacant() const {
    return state == State::VACANT;
  }

  /**
   *
   * @tparam R - random engine type
   * @param params - simulation parameters
   * @param vacant_neighbours - number of vacant neighbouring sites
   * @param dose - irradiation dose applied in the current step
   * @param rand - random engine
   * @return true if the cell is ready for division, false otherwise
   */
  template <typename R>
  __host__ __device__ bool step(const Parameters &params, uint8_t vacant_neighbours, float dose,
                                R &rand) {
    if (isOccupied()) {
      uint8_t alive = cell.updateState(substrates, params, vacant_neighbours);
      if (alive) {
        if (dose > 0) cell.irradiate(dose, params.cell_repair);
        cell.metabolise(substrates, params.metabolism);
        bool cycle_changed = cell.progressClock(params.time_step);
        alive = cell.tryRepair(params.cell_repair, cycle_changed, params.time_step, rand);
        if (alive) {
          return cell.phase == Cell::CyclePhase::D;
        } else {
          state = Site::State::VACANT;
        }
      } else {
        state = Site::State::VACANT;
      }
    }
    return false;
  }
};

}  // namespace emt6ro
#endif  // EMT6RO_SITE_SITE_H_
