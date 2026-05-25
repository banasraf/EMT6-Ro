#ifndef EMT6RO_SIMULATION_SIMULATION_H_
#define EMT6RO_SIMULATION_SIMULATION_H_

#include <utility>
#include "emt6ro/common/random-engine.h"
#include "emt6ro/common/device-buffer.h"
#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"
#include "emt6ro/common/protocol.h"

namespace emt6ro {

// Per-kernel CUDA-event timer totals, accumulated across all calls to
// Simulation::step() since construction or the last resetTimers().
// Populated only when the build flag EMT6RO_TIMING is defined; otherwise
// every field stays at its default-constructed value. The host accessor
// getTimers() is always present so the Python API surface is stable.
struct KernelTimers {
  float findOccupied_ms = 0.f;
  float updateROIs_ms = 0.f;
  float diffuse_ms = 0.f;
  float simulateCells_ms = 0.f;
  float countLiving_ms = 0.f;
  uint64_t n_steps = 0;
};

class Simulation {
 public:
  Simulation(uint32_t batch_size, const Parameters &parameters, uint32_t seed);

#ifdef EMT6RO_TIMING
  ~Simulation();
#endif

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
#ifdef EMT6RO_TIMING
    timers_ = rhs.timers_;
    for (int i = 0; i < 5; ++i) {
      evt_start_[i] = rhs.evt_start_[i];
      evt_stop_[i] = rhs.evt_stop_[i];
    }
    events_inited_ = rhs.events_inited_;
    rhs.events_inited_ = false;  // prevent double-destroy in rhs's dtor
#endif
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

  size_t batchSize() const {
    return batch_size;
  }

  Dims getDims() const {
    return dims;
  }

  // Returns accumulated per-kernel ms since construction or resetTimers().
  // Returns default-constructed KernelTimers (all zeros) in builds without
  // EMT6RO_TIMING.
  KernelTimers getTimers() const {
#ifdef EMT6RO_TIMING
    return timers_;
#else
    return KernelTimers{};
#endif
  }

  void resetTimers() {
#ifdef EMT6RO_TIMING
    timers_ = KernelTimers{};
#endif
  }

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
#ifdef EMT6RO_TIMING
  // Indices: 0=findOccupied 1=updateROIs 2=diffuse 3=simulateCells 4=countLiving.
  KernelTimers timers_{};
  cudaEvent_t evt_start_[5]{};
  cudaEvent_t evt_stop_[5]{};
  bool events_inited_ = false;
#endif
};

}  // namespace emt6ro

#endif  // EMT6RO_SIMULATION_SIMULATION_H_
