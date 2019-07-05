#include <gtest/gtest.h>
#include "emt6ro/cell/cell.h"

namespace emt6ro {
namespace {

const float time_step = 360;

struct TestParams {
  Cell::CyclePhase current_phase;
  Cell::CyclePhase expected_phase;
  Cell::MetabolicMode mode;
  float current_time;
  bool expected_result;
};

class ProgressClockTest : public testing::TestWithParam<TestParams> {
 protected:
  void SetUp() override {
    auto params = GetParam();
    cell.phase = params.current_phase;
    cell.mode = params.mode;
    cell.proliferation_time = params.current_time;
    cell.cycle_times = {1, 2, 3, 4, 5};
  }

  Cell cell{};
};

}  // namespace

TEST_P(ProgressClockTest, ProgressClockTest) {
  auto params = GetParam();
  bool cycle_changed = cell.progressClock(time_step);
  ASSERT_EQ(cycle_changed, params.expected_result);
  ASSERT_EQ(cell.phase, params.expected_phase);
  if (cell.mode == Cell::MetabolicMode::AEROBIC_PROLIFERATION ||
      cell.mode == Cell::MetabolicMode::ANAEROBIC_PROLIFERATION) {
    ASSERT_EQ(cell.proliferation_time, params.current_time + time_step / 3600);
  } else {
    ASSERT_EQ(cell.proliferation_time, params.current_time);
  }
}

INSTANTIATE_TEST_SUITE_P(NoProgress, ProgressClockTest, ::testing::Values(
    TestParams{Cell::CyclePhase::G1, Cell::CyclePhase::G1,
               Cell::MetabolicMode::AEROBIC_QUIESCENCE, 0.95, false},
    TestParams{Cell::CyclePhase::G1, Cell::CyclePhase::G1,
               Cell::MetabolicMode::ANAEROBIC_QUIESCENCE, 0.95, false}));


INSTANTIATE_TEST_SUITE_P(Progress, ProgressClockTest, ::testing::Values(
    TestParams{Cell::CyclePhase::G1, Cell::CyclePhase::S,
               Cell::MetabolicMode::AEROBIC_PROLIFERATION, 0.95, true},
    TestParams{Cell::CyclePhase::S, Cell::CyclePhase::S,
               Cell::MetabolicMode::AEROBIC_PROLIFERATION, 1.85, false},
    TestParams{Cell::CyclePhase::M, Cell::CyclePhase::D,
               Cell::MetabolicMode::ANAEROBIC_PROLIFERATION, 3.95, false}));

}  // namespace emt6ro
