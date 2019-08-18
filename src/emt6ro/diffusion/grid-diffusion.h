#ifndef SRC_EMT6RO_DIFFUSION_GRID_DIFFUSION_H_
#define SRC_EMT6RO_DIFFUSION_GRID_DIFFUSION_H_

#include <cuda_runtime.h>
#include "emt6ro/site/site.h"
#include "emt6ro/common/grid.h"

namespace emt6ro {

__host__ __device__ double paramDiffusion(float val, float coeff, float tau,
                                          float ortho_sum, float diag_sum);

__device__ void diffuse(const GridView<float> &input, GridView<float> &output,
                        float coeff, float time_step);

void batchDiffuse(float *data, Dims dims, size_t batch_size,
                  float coeff, float time_step, uint32_t steps);

void copySubstrates(float *cho, float *ox, float *gi,
                    const Site *sites, Dims dims, uint32_t batch_size);

void copySubstratesBack(Site *sites, const float *cho, const float *ox, const float *gi,
                        Dims dims, uint32_t batch_size);

}  // namespace emt6ro

#endif  // SRC_EMT6RO_DIFFUSION_GRID_DIFFUSION_H_
