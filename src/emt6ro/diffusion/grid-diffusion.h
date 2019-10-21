#ifndef SRC_EMT6RO_DIFFUSION_GRID_DIFFUSION_H_
#define SRC_EMT6RO_DIFFUSION_GRID_DIFFUSION_H_

#include <cuda_runtime.h>
#include "emt6ro/site/site.h"
#include "emt6ro/common/grid.h"

namespace emt6ro {

void findTumorsBoundaries(const GridView<Site> *lattices, ROI *rois, uint32_t batch_size);

void batchDiffuse2(GridView<Site> *lattices, const ROI *rois, Substrates *temp_mem,
                   Dims max_dims, const Substrates &coeffs, const Substrates &ext_levels,
                   uint32_t batch_size, float time_step, uint32_t steps);

}  // namespace emt6ro

#endif  // SRC_EMT6RO_DIFFUSION_GRID_DIFFUSION_H_
