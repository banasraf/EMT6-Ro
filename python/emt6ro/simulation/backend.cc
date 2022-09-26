#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <future>
#include <algorithm>
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/diffusion/diffusion.h"
#include "emt6ro/state/state.h"
#include "emt6ro/common/error.h"

namespace emt6ro {

namespace py = pybind11;

class Experiment {
 public:

 private:
  int device_id;
  int tumors_num; // number of tumors
  int tests_num; // number of tests for each protocol and tumor
  int protocols_num; // number of protocols
  uint32_t simulation_steps;
  uint32_t protocol_resolution;
  device::buffer<Site> tumors_data;
  int64_t protocol_data_size;
  device::Guard device_guard;
  device::unique_ptr<float[]> protocols_data;
  std::vector<Protocol> protocols;
  std::random_device rd{};
  Parameters params;
  Simulation simulation;
  std::future<std::vector<uint32_t>> results_;
  bool running = false;

 public:
  Experiment(const Parameters& params, std::vector<HostGrid<Site>*> tumors,
             int tests_num, int protocols_num, uint32_t sim_steps,
             uint32_t prot_resolution, int device_id)
  : device_id{device_id}
  , tumors_num{static_cast<int>(tumors.size())}
  , tests_num{tests_num}
  , protocols_num{protocols_num}
  , simulation_steps(sim_steps)
  , protocol_resolution(prot_resolution)
  , tumors_data(tests_num * protocols_num * acc_volume(tumors))
  , protocol_data_size{(sim_steps + prot_resolution - 1) / prot_resolution}
  , device_guard{device_id}
  , protocols_data()
  , protocols(protocols_num * tests_num * tumors_num)
  , params(params)
  , simulation(tumors_num * tests_num * protocols_num, params, rd()) {
    device::Guard d_g{device_id};
    float *p_data;
    cudaMallocManaged(&p_data, protocol_data_size * protocols_num * sizeof(float));
    protocols_data = device::unique_ptr<float[]>(p_data, device::Deleter{device_id});
    uint32_t index = 0;
    for (int p = 0; p < protocols_num; ++p) {
      for (auto tumor : tumors) {
        for (int t = 0; t < tests_num; ++t) {
          tumors_data.copyHost(tumor->view().data, tumor->view().dims.vol(), index, simulation.stream());
          index += tumor->view().dims.vol();
        }
      }
    }
    for (int p = 0; p < protocols_num; ++p) {
      Protocol prot{protocol_resolution, simulation_steps,
                    protocols_data.get() + p * protocol_data_size};
      for (int t = 0; t < tumors_num * tests_num; ++t) {
        protocols[p * tumors_num * tests_num + t] = prot;
      }
    }
    simulation.setProtocols(protocols.data());
    reset();
  }

  static int32_t acc_volume(const std::vector<HostGrid<Site>*> &tumors) {
    int size = 0;
    for (auto &t : tumors) size += t->view().dims.vol();
    return size;
  }

  void reset() {
    device::Guard d_g{device_id};
    simulation.reset();
    simulation.setState(tumors_data.data());
    memset(protocols_data.get(), 0 , protocol_data_size * protocols_num * sizeof(float));
  }

  void addIrradiations(const std::vector<std::vector<std::pair<int, float>>> &ps) {
    ENFORCE(ps.size() == protocols_num,
            make_string("Wrong number of protocols. Expected: ", protocols_num));
    device::Guard d_g{device_id};
    for (int i = 0; i < protocols_num; ++i) {
      auto p = protocols[i * tumors_num * tests_num];
      for (auto time_dose : ps[i]) {
        p.closestDose(time_dose.first) += time_dose.second;
      }
    }
  }

  void run(int nsteps) {
    ENFORCE(!running, "Experiment already running.");
    running = true;
    results_ = std::async(std::launch::async, [&, nsteps]() {
      device::Guard d_g{device_id};
      simulation.run(nsteps);
      std::vector<uint32_t> res(tumors_num * tests_num * protocols_num);
      simulation.getResults(res.data());
      return res;
    });
  }

  std::vector<uint32_t> results() {
    device::Guard dg{device_id};
    if (!running)
      throw std::runtime_error("First you have to run the experiment.");
    running = false;
    return results_.get();
  }

  std::vector<HostGrid<Site>> state() {
    device::Guard dg{device_id};
    std::vector<HostGrid<Site>> states;
    size_t bs = simulation.batchSize();
    states.reserve(bs);
    for (size_t s = 0; s < bs; ++s) {
      states.push_back(HostGrid<Site>(simulation.getDims()));
      auto &state = states.back();
      simulation.getData(state.view().data, s);
    }
    return states;
  }

};

static py::array_t<float> getIrradiation(const HostGrid<Site> &state) {
  const float *origin = &state.view().data->cell.irradiation;
  std::array<ssize_t, 2> shape = {state.view().dims.height, state.view().dims.width};
  std::array<ssize_t, 2> stride = {state.view().dims.width * sizeof(Site), sizeof(Site)};
  py::buffer_info buffer(const_cast<float*>(origin), shape, stride, true);
  return py::array(buffer);
}

static py::array_t<Site::State> getOccupancy(const HostGrid<Site> &state) {
  const Site::State *origin = &state.view().data->state;
  std::array<ssize_t, 2> shape = {state.view().dims.height, state.view().dims.width};
  std::array<ssize_t, 2> stride = {state.view().dims.width * sizeof(Site), sizeof(Site)};
  py::buffer_info buffer(const_cast<Site::State*>(origin), shape, stride, true);
  return py::array(buffer);
}

PYBIND11_MODULE(backend, m) {
  PYBIND11_NUMPY_DTYPE(Substrates, cho, ox, gi);
  PYBIND11_NUMPY_DTYPE(Coords, r, c);

  py::enum_<Site::State>(m, "SiteState")
      .value("Vacant", Site::State::VACANT)
      .value("Occupied", Site::State::OCCUPIED)
      .value("Mocked", Site::State::MOCKED)
      .export_values();

  py::class_<Coords>(m, "Coords")
      .def_readwrite("r", &Coords::r)
      .def_readwrite("c", &Coords::c)
      .def("__repr__", [](const Coords &self) {
        return make_string("{r: ", self.r, ", c: ", self.c, "}");
      });

  py::class_<HostGrid<Site>>(m, "TumorState")
      .def("irradiation", &getIrradiation, py::return_value_policy::reference_internal)
      .def("occupancy", &getOccupancy, py::return_value_policy::reference_internal);

  py::class_<Parameters>(m, "Parameters");

  py::class_<Experiment>(m, "_Experiment")
      .def(py::init<const Parameters&, std::vector<HostGrid<Site>*>, int, int, int, int, int>())
      .def("run", &Experiment::run)
      .def("results", &Experiment::results)
      .def("add_irradiations", &Experiment::addIrradiations)
      .def("reset", &Experiment::reset)
      .def("state", &Experiment::state);

  m.def("load_parameters", &Parameters::loadFromJSONFile);

  m.def("load_state", &loadFromFile);

}

}
