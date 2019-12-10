#ifndef EMT6RO_SIMULATION_OLD_DIFFUSION_H
#define EMT6RO_SIMULATION_OLD_DIFFUSION_H

#include "emt6ro/common/grid.h"
#include "emt6ro/parameters/parameters.h"
#include "emt6ro/site/site.h"
#include <vector>
namespace emt6ro {
namespace old {

using ul = unsigned long;

std::pair<std::pair<ul, ul>, ul> findTumor(const GridView<Site> &state);

std::pair<std::pair<ul, ul>, std::pair<ul, ul>> findSubLattice(std::pair<ul, ul> mid,
                                                               ul maxDist,
                                                               const GridView<Site> &state);

std::vector<std::pair<ul, ul>> findBorderSites(ul borderedH, ul borderedW,
                                               ul maxDist);

void batchDiffuse(Site *states, Dims dims, const Parameters &params, int32_t batch_size);

}
}

#endif  // EMT6RO_SIMULATION_OLD_DIFFUSION_H
