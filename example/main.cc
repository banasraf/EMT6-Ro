#include <iostream>
#include <random>
#include "emt6ro/division/cell-division.h"
#include "emt6ro/diffusion/grid-diffusion.h"
#include "emt6ro/simulation/simulation.h"

const uint32_t DIM = 22;
const uint32_t BATCH_SIZE = 10;

using emt6ro::Site;
using emt6ro::GridView;
using emt6ro::Dims;
using emt6ro::Parameters;

class Rand {
 public:
  explicit Rand(uint32_t seed): gen{seed} {}

  float uniform() {
    std::uniform_real_distribution<float> dist(0, 1);
    return dist(gen);
  }

  float normal(const Parameters::NormalDistribution &params) {
    std::normal_distribution<float> dist(params.mean, params.stddev);
    return dist(gen);
  }

 private:
  std::mt19937 gen;
};

static void experiment() {
  auto params = Parameters::loadFromJSONFile("../data/default-parameters.json");
  std::vector<Site> data(DIM*DIM);
  GridView<Site> lattice{data.data(), Dims{DIM, DIM}};
  for (uint32_t r = 0; r < DIM; ++r) {
    for (uint32_t c = 0; c < DIM; ++c) {
      auto &site = lattice(r, c);
      site.substrates = params.external_levels;
      if (r == 0 || r == DIM-1 || c == 0 || c == DIM-1) {
        site.state = Site::State::MOCKED;
      } else {
        site.state = Site::State::VACANT;
      }
    }
  }
  auto &site = lattice(11, 11);
  site.state = Site::State::OCCUPIED;
  Rand rand(123);
  site.cell = emt6ro::createCell(params, rand);
  auto simulation = emt6ro::Simulation::FromSingleHost(lattice, BATCH_SIZE, params, 123);
//  simulation.cellDivision();
  for (uint32_t s = 0; s < 20*12*601; ++s)
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
//      std::cout << (int) lattice2(r, c).state << " ";
//      std::cout << (int) lattice2(r, c).cell.phase << " ";
    }
    std::cout << std::endl;
  }
  std::cout << lattice2(11, 11).substrates.cho << std::endl;
}

static void diffusion_exp() {
  auto f = emt6ro::paramDiffusion(2, 0.7, 0.25, 8, 8);
//  std::cout << f << std::endl;
  std::vector<float> data(25, 2.f);
  auto d_data = emt6ro::device::buffer<float>::fromHost(data.data(), 25);
  emt6ro::batchDiffuse(d_data.data(), Dims{5, 5}, 1, 0.7, 0.25, 24);
  auto data2 = d_data.toHost();
  GridView<float> grid2{data2.get(), Dims{5, 5}};
  for (uint32_t r = 0; r < 5; ++r) {
    for (uint32_t c = 0; c < 5; ++c) {
      std::cout << grid2(r, c) << " ";
    }
    std::cout << std::endl;
  }
}

int main() {
  experiment();
//  diffusion_exp();
  return 0;
}