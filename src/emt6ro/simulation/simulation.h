#ifndef EMT6RO_SIMULATION_SIMULATION_H
#define EMT6RO_SIMULATION_SIMULATION_H

#include "emt6ro/common/random-engine.h"
#include "emt6ro/common/device-buffer.h"
#include "emt6ro/common/grid.h"
#include "emt6ro/site/site.h"

namespace emt6ro {

class Simulation {
 public:

  struct DiffusionData {
    device::buffer<float> cho;
    device::buffer<float> ox;
    device::buffer<float> gi;

    explicit DiffusionData(size_t size): cho(size), ox(size), gi(size) {}
  };

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
  , rand_state(batch_size * CuBlockDimX * CuBlockDimY, seeds.data()) {
    cudaMemcpy(d_params.get(), &params, sizeof(Parameters), cudaMemcpyHostToDevice);
  }

  static Simulation FromSingleHost(const GridView<Site> &lattice, uint32_t batch,
                                   const Parameters &params, uint32_t seed);

  void step();


  void calculateVacantNeighbours();

  void diffuse();

  void simulateCells();

  void cellDivision();

  static const uint32_t CuBlockDimX = 32;
  static const uint32_t CuBlockDimY = 32;

  size_t batch_size;
  Dims dims;
  Parameters params;
  device::unique_ptr<Parameters> d_params;
  device::buffer<Site> data;
  device::buffer<GridView<Site>> lattices;
  DiffusionData diffusion_tmp_data;
  device::buffer<uint8_t> vacant_neighbours;
  CuRandEngineState rand_state;
};

}

#endif //EMT6RO_SIMULATION_SIMULATION_H
