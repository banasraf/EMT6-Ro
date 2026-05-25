// Google Benchmark target for the EMT6-Ro inner loop. Built only with
// -DEMT6RO_BUILD_BENCH=ON. Reports per-step CUDA-event time at varying
// batch_size; pair with -DEMT6RO_TIMING=ON to also get the per-kernel
// breakdown via Simulation::getTimers() (not used here — see the Python
// harness in tools/perf_baseline.py for that).
//
// Data dir is taken from EMT6RO_DATA_DIR (env var), defaulting to
// "../../../data" so the binary built at build/src/bench/ resolves the
// data tree when invoked from that directory.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <vector>

#include "emt6ro/common/protocol.h"
#include "emt6ro/simulation/simulation.h"
#include "emt6ro/state/state.h"

namespace {

std::string DataDir() {
  const char *env = std::getenv("EMT6RO_DATA_DIR");
  return env ? std::string(env) : std::string("../../../data");
}

}  // namespace

static void BM_FullStep(benchmark::State &state) {
  uint32_t batch = static_cast<uint32_t>(state.range(0));
  auto params = emt6ro::Parameters::loadFromJSONFile(DataDir() + "/default-parameters.json");
  auto tumor = emt6ro::loadFromFile(DataDir() + "/tumor-lib/tumor-4.txt", params);
  emt6ro::device::buffer<float> protocol_data(2880);
  std::vector<float> zeros(2880, 0.f);
  cudaMemcpy(protocol_data.data(), zeros.data(),
             zeros.size() * sizeof(float), cudaMemcpyHostToDevice);
  emt6ro::Protocol protocol{300, 144000, protocol_data.data()};
  emt6ro::Simulation sim(batch, params, 42);
  sim.sendData(tumor, protocol, batch);

  // Warmup: force caches + JIT compile.
  for (int i = 0; i < 100; ++i) sim.step();
  cudaDeviceSynchronize();

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  for (auto _ : state) {
    cudaEventRecord(s, sim.stream());
    sim.step();
    cudaEventRecord(e, sim.stream());
    cudaEventSynchronize(e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    state.SetIterationTime(ms / 1000.0f);
  }
  cudaEventDestroy(s);
  cudaEventDestroy(e);

  state.counters["sims_per_sec"] = benchmark::Counter(
      static_cast<double>(batch) * state.iterations(),
      benchmark::Counter::kIsRate);
  state.counters["batch"] = static_cast<double>(batch);
}
BENCHMARK(BM_FullStep)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_MAIN();
