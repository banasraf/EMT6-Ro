#include <gtest/gtest.h>
#include "emt6ro/cell/cell.h"

#define ASSERT_SUBSTRATES_EQ(s1, s2) \
ASSERT_EQ(s1.cho, s2.cho);          \
ASSERT_EQ(s1.ox, s2.ox);          \
ASSERT_EQ(s1.gi, s2.gi)

namespace emt6ro {
namespace {

class MetaboliseTest : public testing::TestWithParam<Cell::MetabolicMode> {
 protected:
  MetaboliseTest() {
    metabolism.aerobic_proliferation = {0.11, 0.12, 0.13};
    metabolism.anaerobic_proliferation = {0.21, 0.22, 0.23};
    metabolism.aerobic_quiescence = {0.31, 0.32, 0.33};
    metabolism.anaerobic_quiescence = {0.41, 0.42, 0.43};
  }

  void SetUp() override {
    cell.mode = GetParam();
    cell.metabolise(site_substrates, metabolism);
  }

  Parameters::Metabolism metabolism{};
  Cell cell{};
  Substrates site_substrates{1, 1, 1};
};

}  // namespace

TEST_P(MetaboliseTest, MetaboliseTest) {
  Substrates base{1, 1, 1};
  switch (cell.mode) {
    case Cell::MetabolicMode::AEROBIC_PROLIFERATION: {
      base -= metabolism.aerobic_proliferation;
      ASSERT_SUBSTRATES_EQ(site_substrates, base);
      break;
    }
    case Cell::MetabolicMode::ANAEROBIC_PROLIFERATION: {
      base -= metabolism.anaerobic_proliferation;
      ASSERT_SUBSTRATES_EQ(site_substrates, base);
      break;
    }
    case Cell::MetabolicMode::AEROBIC_QUIESCENCE: {
      base -= metabolism.aerobic_quiescence;
      ASSERT_SUBSTRATES_EQ(site_substrates, base);
      break;
    }
    case Cell::MetabolicMode::ANAEROBIC_QUIESCENCE: {
      base -= metabolism.anaerobic_quiescence;
      ASSERT_SUBSTRATES_EQ(site_substrates, base);
      break;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(MetaboliseTestAllModes, MetaboliseTest, ::testing::Values(
    Cell::MetabolicMode::AEROBIC_PROLIFERATION,
    Cell::MetabolicMode::ANAEROBIC_PROLIFERATION,
    Cell::MetabolicMode::AEROBIC_QUIESCENCE,
    Cell::MetabolicMode::ANAEROBIC_QUIESCENCE));

}  // namespace emt6ro
