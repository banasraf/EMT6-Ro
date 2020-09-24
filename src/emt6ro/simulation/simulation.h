#ifndef EMT6RO_SIMULATION_SIMULATION_H_
#define EMT6RO_SIMULATION_SIMULATION_H_

#include <utility>
#include "emt6ro/common/random-engine.h"
#include "emt6ro/common/device-buffer.h"
#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"
#include "emt6ro/common/protocol.h"

namespace emt6ro {

class Simulation {
 public:
  Simulation(uint32_t batch_size, const Parameters &parameters, uint32_t seed);

  Simulation& operator=(Simulation &&rhs) {
    if (&rhs == this)
      return *this;
    batch_size = rhs.batch_size;
    params = rhs.params;
    data = std::move(rhs.data);
    protocols = std::move(rhs.protocols);
    filled_samples = rhs.filled_samples;
    lattices = std::move(rhs.lattices);
    rois = std::move(rhs.rois);
    border_masks = std::move(rhs.border_masks);
    occupied = std::move(rhs.occupied);
    rand_state = std::move(rhs.rand_state);
    results = std::move(rhs.results);
    step_ = rhs.step_;
    str = std::move(rhs.str);
    return *this;
  }

  /**
   * Send simulation data to GPU.
   * @param grid - tumor data
   * @param protocol - view over **device** data with irradiation protocol
   * @param multi - number of data slots to fill with the given simulation
   */
  void sendData(const HostGrid<Site> &grid, const Protocol &protocol, uint32_t multi = 1);

  void sendData(const HostGrid<Site> &grid, uint32_t multi = 1);

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
    return str.stream_;
  }

  void step();

  void reset();

  void setState(const Site *state);

  void setProtocols(const Protocol *protocols);

 private:
  void populateLattices();

  void diffuse();

  void simulateCells();

  void updateROIs();

  int simulate_num_threads = 512;

  size_t batch_size;
  Dims dims;
  Parameters params;
  device::buffer<Site> data;
  device::buffer<Protocol> protocols;
  uint32_t filled_samples = 0;
  uint32_t filled_protocols = 0;
  device::buffer<GridView<Site>> lattices;
  device::buffer<ROI> rois;
  device::buffer<uint8_t> border_masks;
  device::buffer<uint32_t> occupied;
  CuRandEngineState rand_state;
  device::buffer<uint32_t> results;
  uint32_t step_ = 0;
  device::Stream str{};
};

}  // namespace emt6ro

#endif  // EMT6RO_SIMULATION_SIMULATION_H_
