#include <iostream>
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/state/state.h"
#include "emt6ro/common/device-buffer.h"
#include <fstream>

const int prot_length = 240;
const int prot_count = 60;
const int N = 1000;
const int BATCH_SIZE = 1000;
const int DIM = 53;

const uint32_t HOUR_STEPS = 600;
const uint32_t SIM_LENGTH = 10 * 24 * HOUR_STEPS;  // 10 days
const uint32_t PROTOCOL_RES = HOUR_STEPS / 2;  // 30 minutes

using emt6ro::Parameters;
namespace device = emt6ro::device;
using emt6ro::Simulation;
using emt6ro::Protocol;

void processResults(const std::vector<uint32_t> &results) {
  for (int p = 0; p < BATCH_SIZE / N; ++p) {
    float avg = 0;
    for (int i = p * N; i < (p + 1) * N; ++i) {
      avg += static_cast<float>(results[i]) / N;
    }
    float var = 0;
    for (int i = p * N; i < (p + 1) * N; ++i) {
      auto r = static_cast<float>(results[i]);
      var += (r - avg) * (r - avg) / N;
    }
    std::cout << avg << " " << var << std::endl;
  }
}

int main() {
  auto params = Parameters::loadFromJSONFile("../data/default-parameters.json");
  auto state = emt6ro::loadFromFile("../data/test_tumor.txt", params);
  auto h_view = state.view();
  for (int r = 0; r < h_view.dims.height; ++r) {
    for (int c = 0; c < h_view.dims.width; ++c) {
      if (!h_view(r, c).isOccupied()) {
        h_view(r, c).substrates.cho = params.external_levels.cho;
        h_view(r, c).substrates.ox = params.external_levels.ox;
      }
    }
  }
  std::vector<std::vector<float>> protocols_data_h(prot_count, std::vector<float>(prot_length));
  std::ifstream protocols_file("../ref-protocols.csv");
  if (!protocols_file.good()) {
    std::cerr << "error" << std::endl;
    return 1;
  }
  for (int p = 0; p < prot_count; ++p) {
    for (int s = 0; s < prot_length; ++s) {
      protocols_file >> protocols_data_h[p][s];
    }
  }
  std::vector<device::buffer<float>> protocols_d;
  for (auto &p: protocols_data_h) {
    protocols_d.push_back(std::move(device::buffer<float>::fromHost(p.data(), p.size())));
  }
  std::random_device rd{};
  auto simulation1 = Simulation({DIM, DIM}, BATCH_SIZE, params, rd());
  int protocol_index = 0;
  std::vector<uint32_t> results(BATCH_SIZE);
  auto prepare_simulation = [&protocol_index, &state, &protocols_d](Simulation &simulation) {
    simulation.reset();
    for (int p = 0; p < BATCH_SIZE / N; ++p) {
      Protocol protocol{PROTOCOL_RES, SIM_LENGTH / 2, protocols_d[protocol_index++].data()};
      simulation.sendData(state, protocol, N);
    }
  };
  for (int series = 0; series < prot_count / (BATCH_SIZE / N); series++) {
    prepare_simulation(simulation1);
    simulation1.run(SIM_LENGTH);
    simulation1.getResults(results.data());
    processResults(results);
    std::cerr << "series " << series << "done" << std::endl;
  }
  return 0;
}