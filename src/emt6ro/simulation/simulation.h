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
  Simulation(uint32_t batch_size, const Parameters &parameters, uint32_t seed);

  /**
   * Send simulation data to GPU.
   * @param grid - tumor data
   * @param protocol - view over **device** data with irradiation protocol
   * @param multi - number of data slots to fill with the given simulation
   */
  void sendData(const HostGrid<Site> &grid, const Protocol &protocol, uint32_t multi = 1);

  /**
   * Run simulation.
   * The whole batch should be filled (with `sendData`) before executing this function.
   * @param nsteps - number of steps
   */
  void run(uint32_t nsteps);

  /**
   * Get tumors's living cells count.
   * @param h_data - output host buffer
   */
  void getResults(uint32_t *h_data);

  /**
   * Get lattice
   * @param h_data - output host buffer
   * @param sample - sample index
   */
  void getData(Site *h_data, uint32_t sample);

  void sync();

  cudaStream_t stream() {
    return stream_;
  }

  void reset();

 private:
  void populateLattices();

  void step();

  void diffuse();

  void simulateCells();

  void cellDivision();

  void updateROIs();

  size_t batch_size;
  Dims dims;
  Parameters params;
  device::unique_ptr<Parameters> d_params;
  device::buffer<Site> data;
  uint32_t filled_samples = 0;
  device::buffer<GridView<Site>> lattices;
  device::buffer<Substrates> diffusion_tmp_data;
  device::buffer<uint8_t> vacant_neighbours;
  device::buffer<ROI> rois;
  device::buffer<Protocol> protocols;
  CuRandEngineState rand_state;
  device::buffer<uint32_t> results;
  uint32_t step_ = 0;
  cudaStream_t stream_;
};

}  // namespace emt6ro

#endif  // EMT6RO_SIMULATION_SIMULATION_H_
