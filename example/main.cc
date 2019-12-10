#include <iostream>
#include <random>
#include <chrono>
#include "emt6ro/common/protocol.h"
#include "emt6ro/diffusion/grid-diffusion.h"
#include "emt6ro/division/cell-division.h"
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/state/state.h"
#include "emt6ro/common/debug.h"

const uint32_t DIM = 53;
const uint32_t BATCH_SIZE = 1000;

using emt6ro::Site;
using emt6ro::GridView;
using emt6ro::Dims;
using emt6ro::Parameters;
using emt6ro::Protocol;
using emt6ro::device::buffer;
using emt6ro::Simulation;
using emt6ro::ROI;
using emt6ro::Coords;

const uint32_t HOUR_STEPS = 600;
const uint32_t SIM_LENGTH = 10 * 24 * HOUR_STEPS;  // 10 days
const uint32_t PROTOCOL_RES = HOUR_STEPS / 2;  // 30 minutes

int dst(int r, int c) {
  return (r - 26) * (r - 26) + (c - 26) * (c - 26);
}

std::ostream &operator<<(std::ostream &stream, const Dims &dims) {
  stream << "Dims{" << dims.height << ", " << dims.width << "}";
  return stream;
}

std::ostream &operator<<(std::ostream &stream, const Coords &coords) {
  stream << "Coords{" << coords.r << ", " << coords.c << "}";
  return stream;
}

std::ostream &operator<<(std::ostream &stream, const ROI &roi) {
  stream << "ROI{" << roi.origin << ", " << roi.dims << "}";
  return stream;
}

static void experiment() {
  auto params = Parameters::loadFromJSONFile("../data/default-parameters.json");
  auto state = emt6ro::loadFromFile("../data/test_tumor.txt", params);
  auto view = state.view();
  for (int r = 1; r < view.dims.height - 1; ++r) {
    for (int c = 1; c < view.dims.width - 1; ++c) {
      if (!view(r, c).isOccupied()) {
        view(r, c).substrates.cho = params.external_levels.cho;
        view(r, c).substrates.ox = params.external_levels.ox;
      }
    }
  }
  std::cout << "ec: " << params.cell_repair.delay_time.exp_coeff << " e: " << params.cell_repair.delay_time.coeff << std::endl;
  std::random_device rd{};
  emt6ro::HostRandEngine rand_eng(rd());
  std::vector<float> protocol_data_h(5 * 24 * HOUR_STEPS / PROTOCOL_RES);  // 5 days protocol
  protocol_data_h[40] =  0.;//1.25;  // 5 Gy on the beginning
  protocol_data_h[54] = 0;//2.5;  // 2.5 Gy - second day, 6 PM
  protocol_data_h[58] = 0;//0.75;  // 2.5 Gy - third day, 6 PM
  protocol_data_h[114] = 0;//0.5;  // 2.5 Gy - third day, 6 PM
  auto protocol_data =
      buffer<float>::fromHost(protocol_data_h.data(), 5 * 24 * HOUR_STEPS / PROTOCOL_RES);
  Protocol protocol{PROTOCOL_RES, SIM_LENGTH / 2, protocol_data.data()};
  auto simulation = Simulation({DIM, DIM}, BATCH_SIZE, params, rd());
  simulation.sendData(state, protocol, BATCH_SIZE);
//  for (int i = 0; i < BATCH_SIZE; ++i) {
//    view(26, 26).cell = emt6ro::createCell(params, rand_eng);
//    view(26, 26).state = Site::State::OCCUPIED;
//    simulation.sendData(state, protocol);
//  }
  std::string str = "err" + std::string(cudaGetErrorString(cudaSuccess)) + "str";
  auto start = std::chrono::steady_clock::now();
  try {
    for (int ss = 0; ss < 48; ++ss) {
      simulation.run(1 * 1 * HOUR_STEPS);
      std::vector<uint32_t> results(BATCH_SIZE);
      simulation.getResults(results.data());
      float avg = 0.f;
      for (auto r: results) {
        avg += static_cast<float>(r) / BATCH_SIZE;
      }
      std::cout << avg << ", " << std::flush;
    }
  } catch (const std::exception&) {
    std::cout << simulation.step_ << std::endl;
  }
  std::vector<uint32_t> results(BATCH_SIZE);
  simulation.getResults(results.data());
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Time elapsed: " << static_cast<float>(duration) / 1000 << " seconds" << std::endl;
  std::cout << "Time per simulation: " << static_cast<float>(duration) / (1000 * BATCH_SIZE)
            << " seconds" << std::endl;
  // Print example tumor
  simulation.getData(view.data, 0);
  for (uint32_t r = 1; r < DIM - 1; ++r) {
    for (uint32_t c = 1; c < DIM - 1; ++c) {
//      std::cout << view(r, c).substrates.cho << " ";
//      if (view(r, c).isOccupied()) std::cout << "● "; else std::cout << "· ";
      if (view(r, c).isOccupied()) std::cout << view(r, c).cell.repair_delay_time << " ";
      else std::cout << "· ";
    }
    std::cout << std::endl;
  }
  std::vector<uint32_t> dists(BATCH_SIZE);
  cudaMemcpy(dists.data(), simulation.max_dist.data(), BATCH_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  std::vector<ROI> rois(BATCH_SIZE);
  cudaMemcpy(rois.data(), simulation.rois.data(), BATCH_SIZE * sizeof(ROI), cudaMemcpyDeviceToHost);
  float avg_h = 0.0;
  float avg_w = 0.0;
  for (auto roi: rois) {
    avg_h += static_cast<float>(roi.dims.height) / BATCH_SIZE;
    avg_w += static_cast<float>(roi.dims.width) / BATCH_SIZE;
  }
  std::cout << "h: " << avg_h << " w: " << avg_w << std::endl;
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
  std::array<float, 4> avg_mode{0, 0, 0, 0};
  std::array<float, 5> avg_cycle{0, 0, 0, 0, 0};
  float cho = 0, ox = 0, gi = 0;
  for (int s = 0; s < BATCH_SIZE; ++s) {
    std::array<int32_t, 4> count_mode{0, 0, 0, 0};
    std::array<int32_t, 5> count_cycle{0, 0, 0, 0, 0};
    simulation.getData(view.data, s);
    for (uint32_t r = 1; r < DIM - 1; ++r) {
      for (uint32_t c = 1; c < DIM - 1; ++c) {
        if (view(r, c).isOccupied()) {
          ++count_mode[static_cast<uint8_t>(view(r, c).cell.mode)];
          ++count_cycle[static_cast<uint8_t>(view(r, c).cell.phase)];
        }
      }
    }
    for (int i = 0; i < 4; ++i) avg_mode[i] += static_cast<float>(count_mode[i]) / BATCH_SIZE;
    for (int i = 0; i < 5; ++i) avg_cycle[i] += static_cast<float>(count_cycle[i]) / BATCH_SIZE;
    cho += view(26, 26).substrates.cho / BATCH_SIZE;
    ox += view(26, 26).substrates.ox / BATCH_SIZE;
    gi += view(26, 26).substrates.gi / BATCH_SIZE;
  }
  std::cout << "mode: ";
  for (auto f: avg_mode) std::cout << f << ", ";
  std::cout << std::endl << "phase: ";
  for (auto f: avg_cycle) std::cout << f << ", ";
  std::cout << std::endl;
  std::cout << "cho: " << cho << " ox: " << ox << " gi: " << gi << std::endl;
}

int main() {
  experiment();
  return 0;
}
