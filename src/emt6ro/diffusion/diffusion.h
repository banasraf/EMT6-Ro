#ifndef EMT6RO_DIFFUSION_DIFFUSION_H_
#define EMT6RO_DIFFUSION_DIFFUSION_H_
#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"

namespace emt6ro {

void findROIs(ROI *rois, uint8_t *border_masks, const GridView<Site> *lattices,
              int32_t batch_size, cudaStream_t stream = nullptr);

void batchDiffusion(GridView<Site> *lattices, const ROI *rois, const uint8_t *border_masks,
                    const Parameters::Diffusion &params, Substrates external_levels, int32_t steps,
                    Dims dims, int32_t batch_size, cudaStream_t stream = nullptr);

}

#endif  // EMT6RO_DIFFUSION_DIFFUSION_H_
