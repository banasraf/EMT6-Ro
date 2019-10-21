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
  Simulation(Dims dims, uint32_t batch_size, const Parameters &parameters,
             const device::buffer<uint32_t> &seeds)
  : batch_size(batch_size)
  , dims(dims)
  , params(parameters)
  , d_params(device::alloc_unique<Parameters>(1))
  , data(batch_size * dims.vol())
  , lattices(batch_size)
  , diffusion_tmp_data(batch_size * dims.vol())
  , vacant_neighbours(batch_size * dims.vol())
  , rois(batch_size)
  , protocols(batch_size)
  , rand_state(batch_size * CuBlockDimX * CuBlockDimY, seeds.data()) {
    cudaMemcpy(d_params.get(), &params, sizeof(Parameters), cudaMemcpyHostToDevice);
  }

  static Simulation FromSingleHost(const GridView<Site> &lattice, uint32_t batch,
                                   const Parameters &params, const Protocol &protocol,
                                   uint32_t seed);

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
  device::buffer<GridView<Site>> lattices;
  device::buffer<Substrates> diffusion_tmp_data;
  device::buffer<uint8_t> vacant_neighbours;
  device::buffer<ROI> rois;
  device::buffer<Protocol> protocols;
  CuRandEngineState rand_state;
  uint32_t step_ = 0;
};

}  // namespace emt6ro

#endif  // EMT6RO_SIMULATION_SIMULATION_H_
