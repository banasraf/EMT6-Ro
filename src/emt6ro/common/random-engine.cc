#include "emt6ro/common/random-engine.h"

namespace emt6ro {

float HostRandEngine::normal(const Parameters::NormalDistribution& params) {
  std::normal_distribution<float> dist(params.mean, params.stddev);
  return dist(gen);
}

float HostRandEngine::uniform() {
  std::uniform_real_distribution<float> dist(0, 1);
  return dist(gen);
}

}  // namespace emt6ro
