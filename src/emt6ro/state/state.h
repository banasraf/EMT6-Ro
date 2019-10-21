#ifndef EMT6RO_SIMULATION_STATE_H
#define EMT6RO_SIMULATION_STATE_H

#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"

namespace emt6ro {

HostGrid<Site> loadFromFile(const std::string &filename, const Parameters &parameters);

}

#endif  // EMT6RO_SIMULATION_STATE_H
