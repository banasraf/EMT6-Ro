#include "emt6ro/state/state.h"
#include <fstream>
#include <cassert>
#include <iostream>

namespace emt6ro {

HostGrid<Site> loadFromFile(const std::string& filename, const Parameters &parameters) {
  std::fstream file(filename);
  assert(file.good());
  int32_t h, w;
  file >> h >> w;
  HostGrid<Site> grid({h + 2, w + 2});
  auto view = grid.view();
  for (int32_t r = 0; r < h + 2; ++r) {
    for (int32_t c = 0; c < w + 2; ++c) {
      auto &site = view(r, c);
      if (r == 0 || r == h - 1 || c == 0 || c == w - 1) {
        site.state = Site::State::MOCKED;
        site.substrates = parameters.external_levels;
      } else {
        int s;
        file >> s;
        file >> site.substrates.cho;
        file >> site.substrates.ox;
        file >> site.substrates.gi;
        if (s) {
          site.state = Site::State::OCCUPIED;
          auto &cell = site.cell;
          file >> cell.time_in_repair;
          file >> cell.irradiation;
          cell.calcDelayTime(parameters.cell_repair);
          file >> cell.proliferation_time;
          file >> cell.cycle_times.g1;
          file >> cell.cycle_times.s;
          file >> cell.cycle_times.g2;
          file >> cell.cycle_times.m;
          file >> cell.cycle_times.d;
          int n;
          file >> n;
          cell.mode = static_cast<Cell::MetabolicMode>(n);
          file >> n;
          cell.phase = static_cast<Cell::CyclePhase>(n);
        } else {
          site.state = Site::State::VACANT;
        }
      }
    }
  }
  return grid;
}

}  // namespace emt6ro

