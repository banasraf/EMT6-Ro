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
  std::vector<HostGrid<Site>> tumors_data;
  int64_t protocol_data_size;
  device::Guard device_guard;
  device::buffer<float> protocols_data;
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
  , tumors_data{}
  , protocol_data_size{(sim_steps + prot_resolution - 1) / prot_resolution}
  , device_guard{device_id}
  , protocols_data(protocol_data_size * protocols_num)
  , params(params)
  , simulation(tumors_num * tests_num * protocols_num, params, rd()) {
    for (auto t: tumors) {
      tumors_data.push_back(*t);
    }
  }

  void run(const std::vector<std::vector<std::pair<int, float>>> &protocols) {
    if (running)
      throw std::runtime_error("Experiment already running.");
    if (protocols.size() != protocols_num)
      throw std::runtime_error("Wrong number of protocols.");
    running = true;
      results_ = std::async(std::launch::async, [&, protocols]() {
      device::Guard d_g{device_id};
      prepareProtocolsData(protocols);
      for (int p = 0; p < protocols_num; ++p) {
        Protocol prot{protocol_resolution, simulation_steps,
                      protocols_data.data() + p * protocol_data_size};
        for (auto &tumor : tumors_data) {
          simulation.sendData(tumor, prot, tests_num);
        }
      }
      simulation.run(simulation_steps);
      std::vector<uint32_t> res(tumors_num * tests_num * protocols_num);
      simulation.getResults(res.data()); 
      simulation = Simulation(tumors_num * tests_num * protocols_num, params, rd());
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

 private:
  void prepareProtocolsData(const std::vector<std::vector<std::pair<int, float>>> &protocols) {   
    std::vector<float> host_protocol(protocol_data_size);
    size_t p_i = 0;
    for (const auto &protocol: protocols) {
      std::fill(host_protocol.begin(), host_protocol.end(), 0);
      for (const auto &irradiation: protocol) {
        auto i = irradiation.first / protocol_resolution;
        host_protocol[i] += irradiation.second;
      }
      protocols_data.copyHost(host_protocol.data(), protocol_data_size, p_i * protocol_data_size,
                              simulation.stream());
      ++p_i;
    }
  }

};

PYBIND11_MODULE(backend, m) {
  PYBIND11_NUMPY_DTYPE(Substrates, cho, ox, gi);
  PYBIND11_NUMPY_DTYPE(Coords, r, c);

  py::class_<Coords>(m, "Coords")
      .def_readwrite("r", &Coords::r)
      .def_readwrite("c", &Coords::c)
      .def("__repr__", [](const Coords &self) {
        return make_string("{r: ", self.r, ", c: ", self.c, "}");
      });

  py::class_<HostGrid<Site>>(m, "TumorState");

  py::class_<Parameters>(m, "Parameters");

  py::class_<Experiment>(m, "_Experiment")
      .def(py::init<const Parameters&, std::vector<HostGrid<Site>*>, int, int, int, int, int>())
      .def("run", &Experiment::run)
      .def("results", &Experiment::results);

  m.def("load_parameters", &Parameters::loadFromJSONFile);

  m.def("load_state", &loadFromFile);

}

}
