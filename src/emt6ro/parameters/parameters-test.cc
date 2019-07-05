#include <gtest/gtest.h>
#include "emt6ro/parameters/parameters.h"
#include "tests/params-for-test.h"

namespace emt6ro {
namespace {

void assertDummySubstrates(const Substrates &substrates) {
  ASSERT_EQ(substrates.cho, 1);
  ASSERT_EQ(substrates.ox, 1);
  ASSERT_EQ(substrates.gi, 1);
}

}  // namespace

TEST(Parameters, ParametersLoading) {
  auto p = Parameters::loadFromJSONFile(TEST_DATA_DIR +
                                        std::string("/dummy-parameters.json"));
  ASSERT_EQ(p.lattice_dims.height, 1);
  ASSERT_EQ(p.lattice_dims.width, 1);
  ASSERT_EQ(p.time_step, 1);
  ASSERT_EQ(p.diffusion_params.time_step, 1);
  assertDummySubstrates(p.diffusion_params.coeffs);
  for (auto value : p.metabolism.values) {
    assertDummySubstrates(value);
  }
  assertDummySubstrates(p.external_levels);
  ASSERT_EQ(p.death_gi, 1);
  ASSERT_EQ(p.quiescence_gi, 1);
  ASSERT_EQ(p.death_gi_production, 1);
  for (auto dist : p.cycle_times.params) {
    ASSERT_EQ(dist.mean, 1);
    ASSERT_EQ(dist.stddev, 1);
  }
  ASSERT_EQ(p.cell_repair.delay_time.coeff, 1);
  ASSERT_EQ(p.cell_repair.delay_time.exp_coeff, 1);
  ASSERT_EQ(p.cell_repair.survival_prob.coeff, 1);
  ASSERT_EQ(p.cell_repair.survival_prob.exp_coeff, 1);
  ASSERT_EQ(p.cell_repair.repair_half_time, 1);
}
}  // namespace emt6ro
