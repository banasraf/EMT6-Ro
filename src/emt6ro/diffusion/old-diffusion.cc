#include "old-diffusion.h"
#include <cmath>
#include <algorithm>
#include <vector>

#define MATLAB_2SQRT2 2.828427124746190
#define MATLAB_1_2SQRT2 0.707106781186548

namespace emt6ro {
namespace old {

using double_p = float;

static double_p paramDiffusion(double_p val, double_p d, double_p tau, double_p orthoSum,
                               double_p diagSum) {
  constexpr double_p HS = MATLAB_2SQRT2;
  constexpr double_p f = 4. + HS;
  return (d * tau * HS) / f * (orthoSum + MATLAB_1_2SQRT2 * diagSum - f * val) + val;
}

std::pair<double_p, double_p> sumNeighbors(ul r, ul c, const GridView<double_p> &values) {
  double_p orthogonalResult = 0.0, diagonalResult = 0.0;
  orthogonalResult += values(r-1, c);
  orthogonalResult += values(r+1, c);
  orthogonalResult += values(r, c-1);
  orthogonalResult += values(r, c+1);

  diagonalResult += values(r-1, c-1);
  diagonalResult += values(r-1, c+1);;
  diagonalResult += values(r+1, c-1);;
  diagonalResult += values(r+1, c+1);;
  return {orthogonalResult, diagonalResult};
}

void numericalDiffusion(ul r, ul c, const GridView<double_p> &choCopy,
                                   const GridView<double_p> &oxCopy, const GridView<double_p> &giCopy,
                                   GridView<double_p> choResult, GridView<double_p> oxResult,
                                   GridView<double_p> giResult,
                                   const Parameters::Diffusion &params) {
  auto paramSums = sumNeighbors(r, c, choCopy);
  choResult(r, c) =
      paramDiffusion(choCopy(r, c), params.coeffs.cho, params.time_step, paramSums.first, paramSums.second);
  paramSums = sumNeighbors(r, c, oxCopy);
  oxResult(r, c) =
      paramDiffusion(oxCopy(r, c), params.coeffs.ox, params.time_step, paramSums.first, paramSums.second);
  paramSums = sumNeighbors(r, c, giCopy);
  giResult(r, c) =
      paramDiffusion(giCopy(r, c), params.coeffs.gi, params.time_step, paramSums.first, paramSums.second);
}

static double_p distance(std::pair<double_p, double_p> p1, std::pair<double_p, double_p> p2) {
  return std::sqrt((p1.first - p2.first) * (p1.first - p2.first) +
                   (p1.second - p2.second) * (p1.second - p2.second));
}

std::pair<std::pair<ul, ul>, ul> findTumor(const GridView<Site> &state) {
  ul minC = state.dims.width;
  ul minR = state.dims.height;
  ul maxC = 0;
  ul maxR = 0;
  for (ul r = 0; r < state.dims.height; ++r) {
    for (ul c = 0; c < state.dims.width; ++c) {
      if (state(r, c).isOccupied()) {
        minC = std::min(c, minC);
        minR = std::min(r, minR);
        maxC = std::max(c, maxC);
        maxR = std::max(r, maxR);
      }
    }
  }
  ul midC = minC + (ul)std::round(0.5 * (maxC - minC));
  ul midR = minR + (ul)std::round(0.5 * (maxR - minR));
  ul maxDist = 0;
  for (ul r = minR; r <= maxR; ++r) {
    for (ul c = minC; c <= maxC; ++c) {
      if (state(r, c).isOccupied()) {
        maxDist = std::max(ul(std::ceil(distance({r, c}, {midR, midC}))), maxDist);
      }
    }
  }
  return {{midR, midC}, maxDist};
}

std::pair<std::pair<ul, ul>, std::pair<ul, ul>> findSubLattice(std::pair<ul, ul> mid,
                                                               ul maxDist,
                                                               const GridView<Site> &state) {
  ul midR, midC;
  std::tie(midR, midC) = mid;
  ul subLatticeR = (midR <= maxDist) ? 0 : midR - maxDist;  // sub-lattice upper row
  ul subLatticeC = (midC <= maxDist) ? 0 : midC - maxDist;  // sub-lattice left column
  ul subLatticeW =                                          // sub-lattice width
      (midC + maxDist >= state.dims.width) ? state.dims.width - subLatticeC
                                         : midC + maxDist - subLatticeC + 1;
  ul subLatticeH =  // sub-lattice height
      (midR + maxDist >= state.dims.height) ? state.dims.height - subLatticeR
                                         : midR + maxDist - subLatticeR + 1;
  return {{subLatticeR, subLatticeC}, {subLatticeH, subLatticeW}};
}

struct LatticeCopy {
  ul borderedH;
  ul borderedW;
  std::array<HostGrid<double_p>, 2> CHO;
  std::array<HostGrid<double_p>, 2> OX;
  std::array<HostGrid<double_p>, 2> GI;

  LatticeCopy(int32_t borderedH, int32_t borderedW)
      : borderedH(borderedH)
      , borderedW(borderedW)
      , CHO{HostGrid<double_p>(Dims{borderedH, borderedW}), HostGrid<double_p>(Dims{borderedH, borderedW})}
      , OX{HostGrid<double_p>(Dims{borderedH, borderedW}), HostGrid<double_p>(Dims{borderedH, borderedW})}
      , GI{HostGrid<double_p>(Dims{borderedH, borderedW}), HostGrid<double_p>(Dims{borderedH, borderedW})}
  {}
};

LatticeCopy copySubLattice(ul borderedH, ul borderedW, ul subLatticeR,
                           ul subLatticeC, const GridView<Site> &state) {
  LatticeCopy copy(borderedH, borderedW);
  auto subLatticeH = borderedH - 4;
  auto subLatticeW = borderedW - 4;
  for (ul r = 0; r < subLatticeH; ++r) {
    for (ul c = 0; c < subLatticeW; ++c) {
      copy.CHO[0].view()(r+2, c+2) = state(subLatticeR + r, subLatticeC + c).substrates.cho;
      copy.OX[0].view()(r+2, c+2) = state(subLatticeR + r, subLatticeC + c).substrates.ox;
      copy.GI[0].view()(r+2, c+2) = state(subLatticeR + r, subLatticeC + c).substrates.gi;
    }
  }
  return copy;
}

void calculateDiffusion(LatticeCopy &copy,
                        const std::vector<std::pair<ul, ul>> &borderSites, ul rounds,
                        Parameters::Diffusion &params,
                        Substrates external_levels) {
  auto borderedW = copy.borderedW;
  auto borderedH = copy.borderedH;
  for (ul i = 0; i < rounds; ++i) {
    for (auto rc : borderSites) {
      auto r = rc.first;
      auto c = rc.second;
      copy.CHO[i % 2].view()(r+1, c+1) = external_levels.cho;
      copy.OX[i % 2].view()(r+1, c+1) = external_levels.ox;
      copy.GI[i % 2].view()(r+1, c+1) = external_levels.gi;
    }
    for (ul r = 0; r < borderedH - 2; ++r) {
      for (ul c = 0; c < borderedW - 2; ++c) {
        numericalDiffusion(r + 1, c + 1, copy.CHO[i % 2].view(), copy.OX[i % 2].view(), copy.GI[i % 2].view(),
                           copy.CHO[(i + 1) % 2].view(), copy.OX[(i + 1) % 2].view(), copy.GI[(i + 1) % 2].view(),
                           params);
      }
    }
  }
}

std::vector<std::pair<ul, ul>> findBorderSites(ul borderedH, ul borderedW,
                                                            ul maxDist) {
  std::vector<std::pair<ul, ul>> borderSites;
  for (ul r = 0; r < borderedH - 2; ++r) {
    for (ul c = 0; c < borderedW - 2; ++c) {
      if (distance({r + 1, c + 1}, {double_p(borderedH - 2) / 2., double_p(borderedW - 2) / 2.}) >=
          maxDist) {
        borderSites.emplace_back(r, c);
      }
    }
  }
  return borderSites;
}

void diffusion(GridView<Site> &state, Parameters::Diffusion params, Substrates external_levels, ul steps) {
  std::pair<ul, ul> tumorMid;
  ul maxLivingDistance;
  std::tie(tumorMid, maxLivingDistance) = findTumor(state);
  auto subLatticeCoords = findSubLattice(tumorMid, maxLivingDistance, state);
  ul subLatticeR, subLatticeC, subLatticeW, subLatticeH;
  std::tie(subLatticeR, subLatticeC) = subLatticeCoords.first;
  std::tie(subLatticeH, subLatticeW) = subLatticeCoords.second;

  // add two sites wide border
  auto borderedW = subLatticeW + 4;
  auto borderedH = subLatticeH + 4;
  auto copy = copySubLattice(borderedH, borderedW, subLatticeR, subLatticeC, state);
  auto borderSites = findBorderSites(borderedH, borderedW, maxLivingDistance);
  ul rounds = ul(std::round(steps));
  calculateDiffusion(copy, borderSites, rounds, params, external_levels);
  for (ul r = 0; r < subLatticeH; ++r) {
    for (ul c = 0; c < subLatticeW; ++c) {
      state(subLatticeR + r, subLatticeC + c).substrates.cho =
          copy.CHO[rounds % 2].view()(r +2, c +2);
      state(subLatticeR + r, subLatticeC + c).substrates.ox =
          copy.OX[rounds % 2].view()(r +2, c +2);
      state(subLatticeR + r, subLatticeC + c).substrates.gi =
          copy.GI[rounds % 2].view()(r +2, c +2);
    }
  }
}

void batchDiffuse(Site *states, Dims dims, const Parameters &params, int32_t batch_size) {
  for (int i = 0; i < batch_size; ++i) {
    GridView<Site> view{states, dims};
    diffusion(view, params.diffusion_params, params.external_levels, 24);
    states += dims.vol();
  }
}

}
}