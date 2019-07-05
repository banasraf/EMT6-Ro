#ifndef SRC_EMT6RO_PARAMETERS_PARAMETERS_H_
#define SRC_EMT6RO_PARAMETERS_PARAMETERS_H_

#include <string>
#include "emt6ro/common/grid.h"
#include "emt6ro/common/substrates.h"

namespace emt6ro {

struct Parameters {
  struct NormalDistribution {
    float mean;
    float stddev;
  };

  struct CycleTimesDistribution {
    union {
      NormalDistribution params[5];
      struct {
        NormalDistribution g1;
        NormalDistribution s;
        NormalDistribution g2;
        NormalDistribution m;
        NormalDistribution d;
      };
    };
  };

  struct Metabolism {
    union {
      Substrates values[4];
      struct {
        Substrates aerobic_proliferation;
        Substrates anaerobic_proliferation;
        Substrates aerobic_quiescence;
        Substrates anaerobic_quiescence;
      };
    };
  };

  struct Diffusion {
    Substrates coeffs;
    float time_step;
  };

  /**
   * @brief Describes an exponential function `f(x) = coeff * exp(exp_coeff * x)`
   */
  struct Exponential {
    float coeff;
    float exp_coeff;
  };

  struct CellRepair {
    Exponential delay_time;
    Exponential survival_prob;
    float repair_half_time;
  };

  Dims lattice_dims;
  float time_step;
  Diffusion diffusion_params;
  Metabolism metabolism;
  Substrates external_levels;
  float death_gi;
  float quiescence_gi;
  float death_gi_production;
  CycleTimesDistribution cycle_times;
  CellRepair cell_repair;

  static Parameters loadFromJSONFile(const std::string &file_name);
};

}  // namespace emt6ro

#endif  // SRC_EMT6RO_PARAMETERS_PARAMETERS_H_
