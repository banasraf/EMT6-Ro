#include "emt6ro/state/state.h"
#include "emt6ro/common/error.h"
#include <fstream>
#include <cassert>
#include <iostream>

namespace emt6ro {

HostGrid<Site> loadFromFile(const std::string& filename, const Parameters &parameters) {
  std::fstream file(filename);
  ENFORCE(file.good(), "Could not open the file: ", filename);
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
          if (cell.irradiation > 0)
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

void saveToFile(const HostGrid<Site> &state, const std::string &filename) {
  std::ofstream file(filename);
  ENFORCE(file.good(), "Could not open the file: ", filename);
  auto view = state.view();
  int32_t h = view.dims.height - 2, w = view.dims.width - 2;
  file << h << "\n" << w << "\n";
  for (int32_t r = 1; r < h + 1; ++r) {
    for (int32_t c = 1; c < w + 1; ++c) {
      const auto &site = view(r, c);
      file << static_cast<int>(site.isOccupied()) << "\n";
      file << site.substrates.cho << "\n" << site.substrates.ox << "\n" 
           << site.substrates.gi  << "\n";
      if (site.isOccupied()) {
        const auto &cell = site.cell;
        file << cell.time_in_repair << "\n";
        file << cell.irradiation << "\n";
        file << cell.proliferation_time << "\n";
        file << cell.cycle_times.g1 << "\n";
        file << cell.cycle_times.s << "\n";
        file << cell.cycle_times.g2 << "\n";
        file << cell.cycle_times.m << "\n";
        file << cell.cycle_times.d << "\n";
        file << static_cast<int>(cell.mode) << "\n";
        file << static_cast<int>(cell.phase) << "\n";
      }
    }
  }
}

}  // namespace emt6ro

