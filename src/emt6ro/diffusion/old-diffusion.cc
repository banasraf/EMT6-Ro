#include "emt6ro/diffusion/old-diffusion.h"
#include <vector>
#include <iostream>
#include <omp.h>

namespace emt6ro {

using double_p = float;
using ul = uint32_t;
  
#define MATLAB_2SQRT2 2.828427124746190
#define MATLAB_1_2SQRT2 0.707106781186548

Substrates paramDiffusion(Substrates val, Substrates d, double_p tau, Substrates orthoSum, Substrates diagSum) {
    constexpr double_p HS = MATLAB_2SQRT2;
    constexpr double_p f = 4. + HS;
    return (d*tau*HS)/f * (orthoSum + diagSum*MATLAB_1_2SQRT2 - val*f) + val;
}

std::pair<Substrates, Substrates> sumNeighbors(ul r, ul c, const GridView<Substrates> &values) {
    Substrates orthogonalResult = {0.f, 0.f, 0.f}, diagonalResult  {0.f, 0.f, 0.f};
    orthogonalResult += values(r-1, c);
    orthogonalResult += values(r+1, c);
    orthogonalResult += values(r, c-1);
    orthogonalResult += values(r, c+1);

    diagonalResult += values(r-1, c-1);
    diagonalResult += values(r+1, c-1);
    diagonalResult += values(r+1, c+1);
    diagonalResult += values(r-1, c+1);
    return {orthogonalResult, diagonalResult};
}

void numericalDiffusion(ul r, ul c, const GridView<Substrates> &in, GridView<Substrates> &out, const Parameters::Diffusion &params) {
    auto paramSums = sumNeighbors(r, c, in);
    out(r, c) = paramDiffusion(in(r, c), params.coeffs, params.time_step, paramSums.first, paramSums.second);
}

double_p distance(std::pair<double_p, double_p> p1, std::pair<double_p, double_p> p2) {
    return std::sqrt((p1.first - p2.first) * (p1.first - p2.first) + (p1.second - p2.second) * (p1.second - p2.second));
}

std::pair<Coords, ul> findTumor(const GridView<Site> &state) {
    ul minC = 53;
    ul minR = 53;
    ul maxC = 0;
    ul maxR = 0;
    
    for (ul r = 0; r < 53; ++r) {
        for (ul c = 0; c < 53; ++c) {
            if (state(r, c).isOccupied()) {
                minC = std::min(c, minC);
                minR = std::min(r, minR);
                maxC = std::max(c, maxC);
                maxR = std::max(r, maxR);
            }
        }
    }
    std::cout << "old min r: " << minR << " c: " << minC << std::endl;
    std::cout << "old max r: " << maxR << " c: " << maxC << std::endl;
    ul midC = minC + (ul) std::round(0.5 * (maxC - minC));
    ul midR = minR + (ul) std::round(0.5 * (maxR - minR));
    ul maxDist = 0;
    for (ul r = minR; r <= maxR; ++r) {
        for (ul c = minC; c <= maxC; ++c) {
            if (state(r, c).isOccupied()) {
                maxDist = std::max(ul(std::ceil(distance({r, c}, {midR, midC}))), maxDist);
            }
        }
    }
    return {Coords{midR, midC}, maxDist};
}

std::pair<std::pair<ul, ul>, std::pair<ul, ul>> findSubLattice(Coords mid, ul maxDist) {
    ul midR = mid.r, midC = mid.c;
    ul subLatticeR = (midR <= maxDist) ? 1 : midR - maxDist; // sub-lattice upper row
    ul subLatticeC = (midC <= maxDist) ? 1 : midC - maxDist; // sub-lattice left column
    ul subLatticeW =                                         // sub-lattice width
            (midC + maxDist >= 51) ? 51 - subLatticeC : midC + maxDist - subLatticeC + 1;
    ul subLatticeH =                                         // sub-lattice height
            (midR + maxDist >= 51) ? 51 - subLatticeR : midR + maxDist - subLatticeR + 1;
    std::cout << "old max dist: " << maxDist << std::endl;
    std::cout << "old mid r: " << midR << " c: " << midC << std::endl; 
    std::cout<< "old r: " << subLatticeR << " c: " << subLatticeC << " h: " << subLatticeH << " w: " << subLatticeW << std::endl;
    return {{subLatticeR, subLatticeC}, {subLatticeH, subLatticeW}};
}

std::array<HostGrid<Substrates>, 2> copySubLattice(ul borderedH, ul borderedW, ul subLatticeR, ul subLatticeC, const GridView<Site> &state) {
    std::array<HostGrid<Substrates>, 2> copyD = {HostGrid<Substrates>({borderedH, borderedW}), HostGrid<Substrates>({borderedH, borderedW})};
    auto copy = copyD[0].view();
    auto subLatticeH = borderedH - 4;
    auto subLatticeW = borderedW - 4;
    for (ul r = 0; r < subLatticeH; ++r) {
        for (ul c = 0; c < subLatticeW; ++c) {
            copy(r+2, c+2) = state(subLatticeR + r, subLatticeC + c).substrates;
        }
    }
    return copyD;
}

void calculateDiffusion(
        std::array<HostGrid<Substrates>, 2> &copyD,
        std::vector<Coords> borderSites,
        ul rounds, const Parameters &params) {
    auto borderedW = copyD[0].view().dims.width;
    auto borderedH = copyD[0].view().dims.height;
    for (ul i = 0; i < rounds; ++i) {
        auto in_copy = copyD[i % 2].view();
        auto out_copy = copyD[(i + 1) % 2].view();
        for (auto rc: borderSites) {
            auto r = rc.r;
            auto c = rc.c;
            in_copy(r+1, c+1) = params.external_levels;
        }
        for (ul r = 0; r < borderedH-2; ++r) {
            for (ul c = 0; c < borderedW-2; ++c) {
                numericalDiffusion(r+1, c+1, in_copy, out_copy, params.diffusion_params);
            }
        }
    }
}

std::vector<Coords> findBorderSites(ul borderedH, ul borderedW, ul maxDist) {
    std::vector<Coords> borderSites;
    for (ul r = 0; r < borderedH-2; ++r) {
        for (ul c = 0; c < borderedW-2; ++c) {
            if (distance({r+1, c+1}, {double_p(borderedH-2) / 2., double_p(borderedW-2) / 2.}) >= maxDist) {
                borderSites.emplace_back(Coords{r, c});
            }
        }
    }
    return borderSites;
}


void diffusion(GridView<Site> state, const Parameters &params, uint32_t rounds) {
    Coords tumorMid;
    ul maxLivingDistance;
    std::tie(tumorMid, maxLivingDistance) = findTumor(state);
    auto subLatticeCoords = findSubLattice(tumorMid, maxLivingDistance);
    ul subLatticeR, subLatticeC, subLatticeW, subLatticeH;
    std::tie(subLatticeR, subLatticeC) = subLatticeCoords.first;
    std::tie(subLatticeH, subLatticeW) = subLatticeCoords.second;

    // add two sites wide border
    auto borderedW = subLatticeW + 4;
    auto borderedH = subLatticeH + 4;
    auto copyD = copySubLattice(borderedH, borderedW, subLatticeR, subLatticeC, state);
    auto copy = copyD[rounds % 2].view();
    auto borderSites = findBorderSites(borderedH, borderedW, maxLivingDistance);
    calculateDiffusion(copyD, borderSites, rounds, params);
    for (ul r = 0; r < subLatticeH; ++r) {
        for (ul c = 0; c < subLatticeW; ++c) {
            state(subLatticeR + r, subLatticeC + c).substrates = copy(r+2, c+2);
        }
    }
}

void oldBatchDiffusion(Site *d_data, Dims dims,
                       const Parameters &params, int32_t batch_size) {
  std::vector<Site> h_data(batch_size * dims.vol());
  cudaMemcpy(h_data.data(), d_data, sizeof(Site) * h_data.size(), cudaMemcpyDeviceToHost);
  #pragma omp parallel
  #pragma omp for
  for (int i = 0; i < batch_size; ++i) {
    GridView<Site> state{h_data.data() + dims.vol() * i, dims};
    diffusion(state, params, 24);
  }
  cudaMemcpy(d_data, h_data.data(), sizeof(Site) * h_data.size(), cudaMemcpyHostToDevice);
}

void oldDiffusion(HostGrid<Site> &state, const Parameters &params, uint32_t steps) {
    diffusion(state.view(), params, steps);
}

} // namespace emt6ro
