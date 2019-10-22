#ifndef EMT6RO_STATISTICS_STATISTICS_H
#define EMT6RO_STATISTICS_STATISTICS_H

#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"

namespace emt6ro {

void countLiving(uint32_t *results, Site *data, Dims dims, uint32_t batch_size);

}

#endif  // EMT6RO_STATISTICS_STATISTICS_H
