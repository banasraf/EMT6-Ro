#ifndef SRC_EMT6RO_SITE_SITE_H_
#define SRC_EMT6RO_SITE_SITE_H_

#include <cstdint>
#include "emt6ro/cell/cell.h"
#include "emt6ro/parameters/parameters.h"

namespace emt6ro {

/// @brief Representation of a site on a simulation lattice
struct Site {
  enum class SiteState: uint8_t {
    VACANT = 0,  //!< no living cancerous cells on a site
    OCCUPIED = 1,  //!< site occupied by living cancerous cells
    MOCKED = 2  //!<  dummy state for border sites
  };

  Substrates substrates;  //!< site's substrates levels
  Cell cell;
  SiteState state;
};

}  // namespace emt6ro
#endif  // SRC_EMT6RO_SITE_SITE_H_
