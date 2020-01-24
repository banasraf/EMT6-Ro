#include <iostream>
#include <random>
#include <chrono>
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

static void experiment() {
  auto params = Parameters::loadFromJSONFile("../data/default-parameters.json");
  auto state = emt6ro::loadFromFile("../data/test_tumor.txt", params);
  std::vector<float> protocol_data_h(5 * 24 * HOUR_STEPS / PROTOCOL_RES);  // 5 days protocol
  protocol_data_h[0] = 1;  // 1 Gy on the beginning
  protocol_data_h[42 * HOUR_STEPS / PROTOCOL_RES] = 2.5;  // 2.5 Gy - second day, 6 PM
  protocol_data_h[66 * HOUR_STEPS / PROTOCOL_RES] = 1.5;  // 1.5 Gy - third day, 6 PM
  auto protocol_data =
      buffer<float>::fromHost(protocol_data_h.data(), 5 * 24 * HOUR_STEPS / PROTOCOL_RES);
  Protocol protocol{PROTOCOL_RES, SIM_LENGTH / 2, protocol_data.data()};
  std::random_device rd{};
  auto simulation = Simulation(BATCH_SIZE, params, rd());
  simulation.sendData(state, protocol, BATCH_SIZE);
  auto start = std::chrono::steady_clock::now();
  simulation.run(10 * 24 * HOUR_STEPS);  // 10 days simulation
  std::vector<uint32_t> results(BATCH_SIZE);
  simulation.getResults(results.data());
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Time elapsed: " << static_cast<float>(duration) / 1000 << " seconds" << std::endl;
  std::cout << "Time per simulation: " << static_cast<float>(duration) / (1000 * BATCH_SIZE)
            << " seconds" << std::endl;
  // Print example tumor
  simulation.getData(state.view().data, 0);
  auto view = state.view();
  for (uint32_t r = 1; r < DIM - 1; ++r) {
    for (uint32_t c = 1; c < DIM - 1; ++c) {
      if (view(r, c).isOccupied()) std::cout << "● "; else std::cout << "· ";
    }
    std::cout << std::endl;
  }
  float avg = 0.f;
  for (auto r: results) {
    avg += static_cast<float>(r) / BATCH_SIZE;
  }
  float var = 0.0;
  for (auto r: results) {
    var += (r - avg) * (r - avg) / float(BATCH_SIZE);
  }
  std::cout << "mean: " << avg << std::endl;
  std::cout << "var: " << var << std::endl;
}

int main() {
  experiment();
  return 0;
}
