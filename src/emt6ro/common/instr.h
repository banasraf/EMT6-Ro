#ifndef EMT6RO_COMMON_INSTR_H_
#define EMT6RO_COMMON_INSTR_H_

// Per-event instrumentation for diagnostic / calibration use.
//
// Tracks the following counters on the GPU side:
//   * repair_decisions: number of times the post-irradiation death check
//                       has fired (i.e. time_in_repair >= repair_delay_time
//                       was reached, regardless of the outcome).
//   * repair_kills: subset of repair_decisions that resulted in cell death.
//   * irradiation_events: number of (cell, dose) applications.
//   * divisions: number of successful per-block divisions.
//   * sum_death_prob: cumulative death probability over all repair decisions
//                     (divide by repair_decisions to get the mean).
//   * irr_hist / tir_hist: 300-bin histograms (0.1 unit per bin, range 0..30)
//                          of the irradiation value and the time-in-repair
//                          at the moment each repair decision fired.
//
// Instrumentation atomicAdds are gated by the CMake option EMT6RO_INSTRUMENT
// (default OFF) — in normal builds the counters exist but are never written
// to, so Python read_instr_counters / read_instr_histograms return all zeros.
// Enable with `-DEMT6RO_INSTRUMENT=ON` for calibration or diagnostic runs.

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace emt6ro {

// __device__ counters. Atomic-incremented from kernels only when the build
// flag EMT6RO_INSTRUMENT is defined. Defined in instr.cu (single TU);
// declared extern everywhere else.
extern __device__ unsigned long long g_instr_repair_decisions;
extern __device__ unsigned long long g_instr_repair_kills;
extern __device__ unsigned long long g_instr_irradiation_events;
extern __device__ unsigned long long g_instr_divisions;
extern __device__ double g_instr_sum_death_prob;

// Histogram: 300 buckets of 0.1 Gy each (0 to 30 Gy).
// Same N for time-in-repair (300 buckets of 0.1 hr each, 0 to 30 hr).
constexpr int INSTR_HIST_N = 300;
constexpr float INSTR_HIST_DX = 0.1f;
extern __device__ unsigned long long g_instr_irr_hist[INSTR_HIST_N];
extern __device__ unsigned long long g_instr_tir_hist[INSTR_HIST_N];

struct InstrCounters {
    unsigned long long repair_decisions;
    unsigned long long repair_kills;
    unsigned long long irradiation_events;
    unsigned long long divisions;
    double sum_death_prob;
};

InstrCounters readInstrCounters();
void resetInstrCounters();

// Returns a flat vector of length 2*INSTR_HIST_N: first N are irr_hist, next N are tir_hist.
std::vector<unsigned long long> readInstrHistograms();

}  // namespace emt6ro

#endif  // EMT6RO_COMMON_INSTR_H_
