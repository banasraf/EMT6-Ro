#ifndef SRC_EMT6RO_SITE_SITE_H_
#define SRC_EMT6RO_SITE_SITE_H_

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
};

}  // namespace emt6ro
#endif  // SRC_EMT6RO_SITE_SITE_H_
