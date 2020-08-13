#ifndef EMT6RO_DIFFUSION_OLD_DIFFUSION_H_
#define EMT6RO_DIFFUSION_OLD_DIFFUSION_H_
#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"

namespace emt6ro {

void oldBatchDiffusion(Site *d_data, Dims dims, 
                       const Parameters &params,  int32_t batch_size);

void oldDiffusion(HostGrid<Site> &state, const Parameters &params, uint32_t steps);

}

#endif  // EMT6RO_DIFFUSION_OLD_DIFFUSION_H_
