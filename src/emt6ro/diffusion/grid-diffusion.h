#ifndef EMT6RO_DIFFUSION_GRID_DIFFUSION_H_
#define EMT6RO_DIFFUSION_GRID_DIFFUSION_H_

#include <cuda_runtime.h>
#include "emt6ro/site/site.h"
#include "emt6ro/common/grid.h"

namespace emt6ro {

/**
 * Find tumors' dimensions and localizations
 * @param lattices - tumors' data
 * @param rois - output
 * @param batch_size
 */
void findTumorsBoundaries(const GridView<Site> *lattices, ROI *rois, uint32_t batch_size);

/**
 * Diffusion of substrates on a given batch of tumor grids
 * @param lattices - tumors' data
 * @param rois - tumor boundaries
 * @param temp_mem - temporary memory for diffusion
 * @param max_dims - maximal tumor size
 * @param coeffs - diffusion coefficients
 * @param ext_levels - external substrate levels
 * @param batch_size
 * @param time_step
 * @param steps - number of diffusion steps - must be even number
 */
void batchDiffuse(GridView<Site> *lattices, const ROI *rois, Substrates *temp_mem,
                   Dims max_dims, const Substrates &coeffs, const Substrates &ext_levels,
                   uint32_t batch_size, float time_step, uint32_t steps);

}  // namespace emt6ro

#endif  // EMT6RO_DIFFUSION_GRID_DIFFUSION_H_
