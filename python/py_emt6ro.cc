#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <future>
#include <algorithm>
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/state/state.h"

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
  device::buffer<float> protocols_data;
  std::random_device rd{};
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
  , protocols_data(protocol_data_size * protocols_num)
  , simulation(tumors_num * tests_num * protocols_num, params, rd()) {
    for (auto t: tumors) {
      tumors_data.push_back(*t);
    }
  }

  void run(const std::vector<std::vector<std::pair<int, float>>> &protocols) {
    device::Guard device_guard{device_id};
    if (running)
      throw std::runtime_error("Experiment already running.");
    if (protocols.size() != protocols_num)
      throw std::runtime_error("Wrong number of protocols.");
    prepareProtocolsData(protocols);
    simulation.reset();
    for (int p = 0; p < protocols_num; ++p) {
      Protocol prot{protocol_resolution, simulation_steps,
                    protocols_data.data() + p * protocol_data_size};
      for (auto &tumor : tumors_data) {
        simulation.sendData(tumor, prot, tests_num);
      }
    }
    results_ = std::async(std::launch::async, [&]() {
      device::Guard d_g{device_id};
      simulation.run(simulation_steps);
      std::vector<uint32_t> res(tumors_num * tests_num * protocols_num);
      simulation.getResults(res.data());
      return res;
    });
    running = true;
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

PYBIND11_MODULE(py_emt6ro, m) {
  py::class_<HostGrid<Site>>(m, "TumorState");

  py::class_<Parameters>(m, "Parameters");

  py::class_<Experiment>(m, "Experiment")
      .def(py::init<const Parameters&, std::vector<HostGrid<Site>*>, int, int, int, int, int>())
      .def("run", &Experiment::run)
      .def("results", &Experiment::results);

  m.def("load_parameters", &Parameters::loadFromJSONFile);

  m.def("load_state", &loadFromFile);

  m.def("add_nums", [](int a, int b) {return a + b; });

  m.def("something", [](std::vector<HostGrid<Site>*> &states) {
    for (auto state: states)
      std::cout << state->view().dims.width << std::endl;
  });

}

}