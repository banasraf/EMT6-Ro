#ifndef EMT6RO_SIMULATION_SIMULATION_H_
#define EMT6RO_SIMULATION_SIMULATION_H_

#include "emt6ro/common/random-engine.h"
#include "emt6ro/common/device-buffer.h"
#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"
#include "emt6ro/common/protocol.h"

namespace emt6ro {

class Simulation {
 public:
  Simulation(Dims dims, uint32_t batch_size, const Parameters &parameters, uint32_t seed);

  void sendData(const HostGrid<Site> &grid, const Protocol &Protocol, uint32_t multi = 1);

  void step();

  void diffuse();

  void simulateCells();

  void cellDivision();

  void updateROIs();

  void run(uint32_t nsteps);

  void getResults(uint32_t *h_data);

  void getData(Site *h_data, uint32_t sample);

 private:
  void populateLattices();

  size_t batch_size;
  Dims dims;
  Parameters params;
  device::unique_ptr<Parameters> d_params;
  device::buffer<Site> data;
  uint32_t filled_samples;
  device::buffer<GridView<Site>> lattices;
  device::buffer<Substrates> diffusion_tmp_data;
  device::buffer<uint8_t> vacant_neighbours;
  device::buffer<ROI> rois;
  device::buffer<Protocol> protocols;
  device::buffer<uint32_t> seeds;
  CuRandEngineState rand_state;
  device::buffer<uint32_t> results;
  uint32_t step_ = 0;
};

}  // namespace emt6ro

#endif  // EMT6RO_SIMULATION_SIMULATION_H_
