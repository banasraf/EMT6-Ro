#include <gtest/gtest.h>
#include "emt6ro/cell/cell.h"

namespace emt6ro {
namespace {

// when mode doesn't matter
const Cell::MetabolicMode MOCK_MODE = Cell::MetabolicMode::AEROBIC_QUIESCENCE;

struct TestParam {
  Substrates levels;
  uint8_t vacant_neighbors;
  float g1_time;
  Cell::CyclePhase phase;
  float proliferation_time;
  bool result_alive;
  Cell::MetabolicMode result_mode;
};

class UpdateStateTest : public testing::TestWithParam<TestParam> {
 protected:
  UpdateStateTest(): params{}, cell{} {}

  void SetUp() override {
    params.metabolism.aerobic_proliferation = {21, 9, 0};
    params.metabolism.anaerobic_proliferation = {54, 0, -3e-3};
    params.metabolism.aerobic_quiescence = {18, 8, 0};
    params.metabolism.anaerobic_quiescence = {45, 0, -3e-5};
    params.death_gi = 5e-2;
    params.quiescence_gi = 2e-2;
    params.time_step = 1;

    cell.phase = GetParam().phase;
    cell.cycle_times.g1 = GetParam().g1_time;
    cell.proliferation_time = GetParam().proliferation_time;
  }

  Parameters params;
  Cell cell;
};

}  // namespace

TEST_P(UpdateStateTest, UpdateStateTest) {
  bool result = cell.updateState(GetParam().levels, params, GetParam().vacant_neighbors);
  ASSERT_EQ(result, GetParam().result_alive);
  if (result)
    ASSERT_EQ(cell.mode, GetParam().result_mode);
}

INSTANTIATE_TEST_SUITE_P(QuickDeath, UpdateStateTest, ::testing::Values(
    TestParam{{17, 10, 0}, 0, 100, Cell::CyclePhase::G1, 50, false, MOCK_MODE},
    TestParam{{18, 10, 6e-2}, 0, 100, Cell::CyclePhase::G1, 50, false, MOCK_MODE}));

INSTANTIATE_TEST_SUITE_P(G1_S_stopping, UpdateStateTest, ::testing::Values(
    TestParam{{18, 10, 0}, 0, 20, Cell::CyclePhase::M, 19, false, MOCK_MODE},
    TestParam{{44, 7, 0}, 0, 20, Cell::CyclePhase::G1, 19, false, MOCK_MODE},
    TestParam{{45, 7, 0}, 0, 20, Cell::CyclePhase::G2, 19, true,
              Cell::MetabolicMode::ANAEROBIC_QUIESCENCE},
    TestParam{{18, 8, 0}, 0, 20, Cell::CyclePhase::G2, 19, true,
              Cell::MetabolicMode::AEROBIC_QUIESCENCE}));

INSTANTIATE_TEST_SUITE_P(GI_quiescence, UpdateStateTest, ::testing::Values(
    TestParam{{18, 10, 3e-2}, 0, 20, Cell::CyclePhase::M, 15, false, MOCK_MODE},
    TestParam{{44, 7, 3e-2}, 0, 20, Cell::CyclePhase::G1, 15, false, MOCK_MODE},
    TestParam{{45, 7, 3e-2}, 0, 20, Cell::CyclePhase::G2, 15, true,
              Cell::MetabolicMode::ANAEROBIC_QUIESCENCE},
    TestParam{{18, 8, 3e-2}, 0, 20, Cell::CyclePhase::G2, 15, true,
              Cell::MetabolicMode::AEROBIC_QUIESCENCE}));

INSTANTIATE_TEST_SUITE_P(ProliferationFail, UpdateStateTest, ::testing::Values(
    TestParam{{18, 8, 0}, 0, 20, Cell::CyclePhase::M, 15, false, MOCK_MODE},
    TestParam{{44, 7, 0}, 0, 20, Cell::CyclePhase::G1, 15, false, MOCK_MODE},
    TestParam{{45, 7, 0}, 0, 20, Cell::CyclePhase::G2, 15, true,
              Cell::MetabolicMode::ANAEROBIC_QUIESCENCE},
    TestParam{{18, 8, 0}, 0, 20, Cell::CyclePhase::G2, 15, true,
              Cell::MetabolicMode::AEROBIC_QUIESCENCE}));

INSTANTIATE_TEST_SUITE_P(AnaerobicProliferation, UpdateStateTest, ::testing::Values(
    TestParam{{54, 7, 0}, 0, 20, Cell::CyclePhase::G1, 15, true,
              Cell::MetabolicMode::ANAEROBIC_PROLIFERATION}));

INSTANTIATE_TEST_SUITE_P(TryAerobicProliferation, UpdateStateTest, ::testing::Values(
    TestParam{{21, 9, 0}, 0, 20, Cell::CyclePhase::G1, 15, true,
              Cell::MetabolicMode::AEROBIC_PROLIFERATION},
    TestParam{{21, 8, 0}, 0, 20, Cell::CyclePhase::G1, 15, true,
              Cell::MetabolicMode::AEROBIC_QUIESCENCE},
    TestParam{{21, 8, 0}, 0, 20, Cell::CyclePhase::M, 15, false, MOCK_MODE}));

}  // namespace emt6ro
