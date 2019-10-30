#ifndef EMT6RO_STATISTICS_STATISTICS_H_
#define EMT6RO_STATISTICS_STATISTICS_H_

#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"

namespace emt6ro {

void countLiving(uint32_t *results, Site *data, Dims dims, uint32_t batch_size,
                 cudaStream_t stream = nullptr);

}  // namespace emt6ro

#endif  // EMT6RO_STATISTICS_STATISTICS_H_
