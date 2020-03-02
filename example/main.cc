#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include "emt6ro/common/protocol.h"
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/state/state.h"

const uint32_t DIM = 53;
const uint32_t BATCH_SIZE = 1000;

using emt6ro::Site;
using emt6ro::GridView;
using emt6ro::Dims;
using emt6ro::Parameters;
using emt6ro::Protocol;
using emt6ro::device::buffer;
using emt6ro::Simulation;

const uint32_t HOUR_STEPS = 600;
const uint32_t SIM_LENGTH = 10 * 24 * HOUR_STEPS;  // 10 days
const uint32_t PROTOCOL_RES = HOUR_STEPS / 2;  // 30 minutes

enum class Feature: unsigned {
  TOTAL_CELLS,
  G1,
  S,
  G2M,
  CHO,
  OX,
  GI
};

class Timelapse {
  std::vector<double> data;
  unsigned tumors, steps, tests;

 public:
  Timelapse(unsigned tumors, unsigned steps, unsigned tests)
  : data(tumors * steps * tests * 7)
  , tumors(tumors)
  , steps(steps)
  , tests(tests) {}

  double &val(unsigned tumor, unsigned step, unsigned test, Feature f) {
    return data[(unsigned)f + 7 * (test + tests * (step + steps * tumor))];
  }

  void dump_tumor(unsigned tumor, const std::string &filename) {
    std::ofstream ofile(filename);
    for (uint32_t test = 0; test < tests; ++test) {
      for (uint32_t step = 0; step < steps; ++step) {
        for (unsigned f = 0; f < 7; ++f) {
          ofile << val(tumor, step, test, static_cast<Feature>(f)) << " ";
        }
        ofile << std::endl;
      }
    }
  }
};

void add_to_timelapse(Simulation &simulation, Timelapse &timelapse, unsigned t) {
  emt6ro::HostGrid<Site> state(Dims{53, 53});
  for (uint32_t tumor = 0; tumor < 10; ++tumor) {
    for (uint32_t test = 0; test < 100; ++test) {
      simulation.getData(state.view().data, tumor * 100 + test);
      auto view = state.view();
      uint32_t phase_count[] = {0, 0, 0, 0, 0};
      uint32_t alive_count = 0;
      for (int32_t r = 0; r < view.dims.height; ++r) {
        for (int32_t c = 0; c < view.dims.width; ++c) {
          if (view(r, c).isOccupied()) {
            ++alive_count;
            ++phase_count[(uint8_t)view(r, c).cell.phase];
          }
        }
      }
      timelapse.val(tumor, t, test, Feature::TOTAL_CELLS) = alive_count;
      timelapse.val(tumor, t, test, Feature::G1) = phase_count[0];
      timelapse.val(tumor, t, test, Feature::S) = phase_count[1];
      timelapse.val(tumor, t, test, Feature::G2M) = phase_count[2] + phase_count[3];
      timelapse.val(tumor, t, test, Feature::CHO) = view(26, 26).substrates.cho;
      timelapse.val(tumor, t, test, Feature::OX) = view(26, 26).substrates.ox;
      timelapse.val(tumor, t, test, Feature::GI) = view(26, 26).substrates.gi;
    }
  }
}

static void experiment() {
  auto params = Parameters::loadFromJSONFile("../data/default-parameters.json");
  std::vector<emt6ro::HostGrid<Site>> states;
  for (int i = 1; i <= 10; ++i) {
    std::cerr << "../data/tumor-lib/tumor-" + std::to_string(i) + ".txt" << std::endl;
    states.emplace_back(
        emt6ro::loadFromFile("../data/tumor-lib/tumor-" + std::to_string(i) + ".txt", params));
//    auto state = states.back().view();
//    for (uint32_t r = 1; r < 52; ++r) {
//      for (uint32_t c = 1; c < 52; ++c) {
//        if (state(r, c).isVacant()) {
//          state(r, c).substrates.cho = params.external_levels.cho;
//          state(r, c).substrates.ox = params.external_levels.ox;
//        }
//      }
//    }
  }

  std::vector<float> protocol_data_h(5 * 24 * HOUR_STEPS / PROTOCOL_RES + 1);  // 5 days protocol
//  protocol_data_h[0] = 1.25;
//  protocol_data_h[17 * HOUR_STEPS / PROTOCOL_RES] = 1.25;
//  protocol_data_h[34 * HOUR_STEPS / PROTOCOL_RES] = 1.25;
//  protocol_data_h[51 * HOUR_STEPS / PROTOCOL_RES] = 1.25;
//  protocol_data_h[68 * HOUR_STEPS / PROTOCOL_RES] = 1.25;
//  protocol_data_h[85 * HOUR_STEPS / PROTOCOL_RES] = 1.25;
//  protocol_data_h[103 * HOUR_STEPS / PROTOCOL_RES] = 1.25;
//  protocol_data_h[120 * HOUR_STEPS / PROTOCOL_RES] = 1.25;
//
  protocol_data_h[0] = 2;
  protocol_data_h[18 * HOUR_STEPS / PROTOCOL_RES] = 2;
  protocol_data_h[36 * HOUR_STEPS / PROTOCOL_RES] = 2;
  protocol_data_h[54 * HOUR_STEPS / PROTOCOL_RES] = 2;
  protocol_data_h[72 * HOUR_STEPS / PROTOCOL_RES] = 2;
  auto protocol_data =
      buffer<float>::fromHost(protocol_data_h.data(), 5 * 24 * HOUR_STEPS / PROTOCOL_RES + 1);
  Protocol protocol{PROTOCOL_RES, SIM_LENGTH / 2 + 1, protocol_data.data()};
  std::random_device rd{};
  auto simulation = Simulation(BATCH_SIZE, params, rd());
  for (int i = 0; i < 10; ++i) {
    simulation.sendData(states[i], protocol, BATCH_SIZE/10);
  }
  Timelapse timelapse(10, 10 * 24 * 60 + 1, 100);
  add_to_timelapse(simulation, timelapse, 0);
  for (int t = 0; t < 10 * 24 * 60; ++t) {
    simulation.run(HOUR_STEPS / 60);
    add_to_timelapse(simulation, timelapse, t+1);
    if (t % 60 == 0) std::cerr << (float)(t/60) << std::endl;
  }
  for (unsigned tumor = 0; tumor < 10; ++tumor) {
    timelapse.dump_tumor(tumor, "tl2_outs/t" + std::to_string(tumor + 1) + ".txt");
  }
}

int main() {
  experiment();
  return 0;
}
