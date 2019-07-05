#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "emt6ro/parameters/parameters.h"

namespace emt6ro {

using nlohmann::json;

namespace {

json parseFile(const std::string &file_name) {
  std::ifstream ifs(file_name);
  return json::parse(ifs);
}

Dims readDims(const json &obj) {
  return {obj["height"].get<uint32_t>(), obj["width"].get<uint32_t>()};
}

Substrates readSubstrates(const json &obj) {
  return {obj["cho"].get<float>(), obj["ox"].get<float>(), obj["gi"].get<float>()};
}

Parameters::Diffusion readDiffusion(const json &obj) {
  return {readSubstrates(obj["coeffs"]), obj["time_step"].get<float>()};
}

Parameters::Metabolism readMetabolism(const json &obj) {
  Parameters::Metabolism result{};
  result.aerobic_proliferation = readSubstrates(obj["aerobic_proliferation"]);
  result.anaerobic_proliferation = readSubstrates(obj["anaerobic_proliferation"]);
  result.aerobic_quiescence = readSubstrates(obj["aerobic_quiescence"]);
  result.anaerobic_quiescence = readSubstrates(obj["anaerobic_quiescence"]);
  return result;
}

Parameters::NormalDistribution readNormalDistribution(const json &obj) {
  return {obj["mean"].get<float>(), obj["stddev"].get<float>()};
}

Parameters::CycleTimesDistribution readCycleTimesDistribution(const json &obj) {
  Parameters::CycleTimesDistribution result{};
  result.g1 = readNormalDistribution(obj["g1"]);
  result.s = readNormalDistribution(obj["s"]);
  result.g2 = readNormalDistribution(obj["g2"]);
  result.m = readNormalDistribution(obj["m"]);
  result.d = readNormalDistribution(obj["d"]);
  return result;
}

Parameters::Exponential readExponential(const json &obj) {
  return {obj["coeff"].get<float>(), obj["exp_coeff"].get<float>()};
}

Parameters::CellRepair readCellRepair(const json &obj) {
  return {
    readExponential(obj["delay_time"]),
    readExponential(obj["survival_prob"]),
    obj["repair_half_time"].get<float>()
        };
}

}  // namespace

Parameters Parameters::loadFromJSONFile(const std::string& file_name) {
  const json obj = parseFile(file_name);
  Parameters result{};
  result.lattice_dims = readDims(obj["lattice_dims"]);
  result.time_step = obj["time_step"].get<float>();
  result.diffusion_params = readDiffusion(obj["diffusion_params"]);
  result.metabolism = readMetabolism(obj["metabolism"]);
  result.external_levels = readSubstrates(obj["external_levels"]);
  result.death_gi = obj["death_gi"].get<float>();
  result.quiescence_gi = obj["quiescence_gi"].get<float>();
  result.death_gi_production = obj["death_gi_production"].get<float>();
  result.cycle_times = readCycleTimesDistribution(obj["cycle_times"]);
  result.cell_repair = readCellRepair(obj["cell_repair"]);
  return result;
}

}  // namespace emt6ro
