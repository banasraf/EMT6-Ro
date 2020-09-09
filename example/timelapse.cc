#include <iostream>
#include <random>
#include <chrono>
#include "emt6ro/common/protocol.h"
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/state/state.h"

const uint32_t DIM = 53;

using emt6ro::Site;
using emt6ro::Cell;
using emt6ro::GridView;
using emt6ro::Dims;
using emt6ro::Parameters;
using emt6ro::Protocol;
using emt6ro::device::buffer;
using emt6ro::Simulation;

const uint32_t HOUR_STEPS = 600;
const uint32_t SIM_LENGTH = 10 * 24 * HOUR_STEPS;  // 10 days
const uint32_t PROTOCOL_RES = HOUR_STEPS / 2;  // 30 minutes

class Timelapse {
  using tl_t = std::vector<std::array<float, 13>>;
 public:
  Timelapse(int bs, int steps) : data(bs, tl_t(steps)) {}

  void addStep(const GridView<Site> &state, int sample, int step) {
    auto &row = data[sample][step];
    row[0] = state(26, 26).substrates.cho;
    row[1] = state(26, 26).substrates.ox;
    row[2] = state(26, 26).substrates.gi;
    for (int i = 3; i < 13; ++i) row[i] = 0;
    for (int r = 1; r < 52; ++r) {
      for (int c = 1; c < 52; ++c) {
        const Site &site = state(r, c);
        if (site.isOccupied()) {
          row[3] += 1;
          const Cell &cell = site.cell;
          row[static_cast<uint8_t>(cell.mode) + 4] += 1;
          row[static_cast<uint8_t>(cell.phase) + 8] += 1;
        }
      }
    }
  }

  void print() {
    for (tl_t &sample : data) {
      for (auto &row : sample) {
        for (float v : row) {
          std::cout << v << ", ";
        }
        std::cout << "\n";
      }
    }
  }

  std::vector<tl_t> data;
};

static void experiment(uint32_t batch_size) {
  uint32_t BATCH_SIZE = batch_size;
  auto params = Parameters::loadFromJSONFile("data/default-parameters.json");
  auto state = emt6ro::loadFromFile("data/tumor-lib/tumor-4.txt", params);
  std::vector<float> protocol_data_h(10 * 24 * HOUR_STEPS / PROTOCOL_RES);  // 5 days protocol
  protocol_data_h[3601 / PROTOCOL_RES] = 2.5;  
  protocol_data_h[11701 / PROTOCOL_RES] = 0.5;  
  protocol_data_h[16501  / PROTOCOL_RES] = 2;  
  protocol_data_h[17101  / PROTOCOL_RES] = 2.25;  
  protocol_data_h[43501  / PROTOCOL_RES] = 0.5;  
  auto protocol_data =
      buffer<float>::fromHost(protocol_data_h.data(), 10 * 24 * HOUR_STEPS / PROTOCOL_RES);
  Protocol protocol{PROTOCOL_RES, SIM_LENGTH, protocol_data.data()};
  std::random_device rd{};
  Simulation simulation(BATCH_SIZE, params, rd());
  simulation.sendData(state, protocol, BATCH_SIZE);
  Timelapse timelapse(batch_size, 2401);
  for (int s = 0; s < batch_size; ++s) {
      simulation.getData(state.view().data, s);
      timelapse.addStep(state.view(), s, 0);
    }
  for (int t = 1; t < 2401; ++t) {
    simulation.run(1);
    for (int s = 0; s < batch_size; ++s) {
      simulation.getData(state.view().data, s);
      timelapse.addStep(state.view(), s, t);
    }
  }
  timelapse.print();
}

int main(int argc, char **argv) {
  int batch_size = std::atoi(argv[1]);
  experiment(batch_size);
  return 0;
}
