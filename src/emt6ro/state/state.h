#ifndef EMT6RO_STATE_STATE_H_
#define EMT6RO_STATE_STATE_H_

#include <string>
#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"

namespace emt6ro {

HostGrid<Site> loadFromFile(const std::string &filename, const Parameters &parameters);

}

#endif  // EMT6RO_STATE_STATE_H_
