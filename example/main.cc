#include <iostream>
#include <random>
#include "emt6ro/division/cell-division.h"
#include "emt6ro/diffusion/grid-diffusion.h"
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/common/protocol.h"
#include "emt6ro/state/state.h"

/*
 * wczytywanie
 * eksperymenty
 */

const uint32_t DIM = 53;
const uint32_t BATCH_SIZE = 500;

using emt6ro::Site;
using emt6ro::GridView;
using emt6ro::Dims;
using emt6ro::Parameters;
using emt6ro::Protocol;
using emt6ro::device::buffer;



static void experiment() {
  auto params = Parameters::loadFromJSONFile("../data/default-parameters.json");
  auto state = emt6ro::loadFromFile("../data/test_tumor.txt", params);
  std::vector<float> protocol_data_h(5 * 24 * 2);
  protocol_data_h[0] = 5;
  protocol_data_h[2 * 24 * 2] = 2.5;
  protocol_data_h[2 * 24 * 2 + 12 * 2] = 2.5;
  auto protocol_data = buffer<float>::fromHost(protocol_data_h.data(), 5 * 24 * 2);
  Protocol protocol{300, 5 * 24 * 2 * 300, protocol_data.data()};
  std::random_device rd{};
  auto simulation = emt6ro::Simulation::FromSingleHost(state.view(), BATCH_SIZE, params, protocol, rd());
//  simulation.cellDivision();
  for (uint32_t s = 0; s < 10 * 24 * 600; ++s)
    simulation.step();
  auto data2 = simulation.data.toHost();
  GridView<Site> lattice2{data2.get() + Dims{DIM, DIM}.vol() * 5, Dims{DIM, DIM}};
  for (uint32_t r = 1; r < DIM - 1; ++r) {
    for (uint32_t c = 1; c < DIM - 1; ++c) {
//      std::cout << (int) lattice2(r, c).meta << " ";
//      std::cout << lattice2(r, c).cell.cycle_times.d << " ";
//      if (lattice2(r, c).isOccupied())
//        std::cout << (int) lattice2(r, c).cell.mode << " ";
//      else
//        std::cout << "· ";
//      std::cout << lattice2(r, c).cell.proliferation_time << " ";
//      std::cout << lattice2(r, c).substrates.gi << " ";
      if (lattice2(r, c).isOccupied()) std::cout << "● "; else std::cout << "· ";
//      if (lattice2(r, c).isOccupied()) std::cout << lattice2(r, c).cell.time_in_repair; else std::cout << "· ";
//      std::cout << (int) lattice2(r, c).state << " ";
//      std::cout << (int) lattice2(r, c).cell.phase << " ";
    }
    std::cout << std::endl;
  }
  std::vector<float> ress(BATCH_SIZE);
  float avg = 0;
  for (int i = 0; i < BATCH_SIZE; ++i) {
    int living = 0;
    GridView<Site> l{data2.get() + Dims{DIM, DIM}.vol() * i, Dims{DIM, DIM}};
    for (uint32_t r = 1; r < DIM - 1; ++r)
      for (uint32_t c = 1; c < DIM - 1; ++c)
        if (l(r, c).isOccupied()) living++;
    ress[i] = living;
    avg += float(living) / float(BATCH_SIZE);
  }
  float var = 0.0;
  for (auto r: ress) {
    var += (r - avg) * (r - avg) / float(BATCH_SIZE);
  }
  std::cout << avg << std::endl;
  std::cout << var << std::endl;
}

//static void diffusion_exp() {
//  auto params = Parameters::loadFromJSONFile("../data/default-parameters.json");
//  std::vector<Site> data(DIM*DIM);
//  GridView<Site> lattice{data.data(), Dims{DIM, DIM}};
//  for (uint32_t r = 0; r < DIM; ++r) {
//    for (uint32_t c = 0; c < DIM; ++c) {
//      auto &site = lattice(r, c);
//      site.substrates = params.external_levels;
//      if (r == 0 || r == DIM-1 || c == 0 || c == DIM-1) {
//        site.state = Site::State::MOCKED;
//      } else {
//        site.state = Site::State::VACANT;
//      }
//    }
//  }
//  auto &site = lattice(26, 26);
//  site.state = Site::State::OCCUPIED;
//  Rand rand(123);
//  site.cell = emt6ro::createCell(params, rand);
//  auto simulation = emt6ro::Simulation::FromSingleHost(lattice, BATCH_SIZE, params, 123);
//  simulation.updateROIs();
//  auto rois = simulation.rois.toHost();
//  for (int32_t i = 0; i < BATCH_SIZE; ++i) {
//    std::cout << "r: " << rois[i].origin.r << " c: " << rois[i].origin.c << " h: "
//    << rois[i].dims.height << " w: " << rois[i].dims.width << std::endl;
//  }
//}

int main() {
  experiment();
//  diffusion_exp();
  return 0;
}