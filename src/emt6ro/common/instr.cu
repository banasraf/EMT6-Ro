#include "emt6ro/common/instr.h"
#include <cuda_runtime.h>

namespace emt6ro {

__device__ unsigned long long g_instr_repair_decisions = 0;
__device__ unsigned long long g_instr_repair_kills = 0;
__device__ unsigned long long g_instr_irradiation_events = 0;
__device__ unsigned long long g_instr_divisions = 0;
__device__ double g_instr_sum_death_prob = 0.0;
__device__ unsigned long long g_instr_irr_hist[INSTR_HIST_N] = {0};
__device__ unsigned long long g_instr_tir_hist[INSTR_HIST_N] = {0};

InstrCounters readInstrCounters() {
    InstrCounters host{};
    cudaMemcpyFromSymbol(&host.repair_decisions, g_instr_repair_decisions,
                         sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&host.repair_kills, g_instr_repair_kills,
                         sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&host.irradiation_events, g_instr_irradiation_events,
                         sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&host.divisions, g_instr_divisions,
                         sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&host.sum_death_prob, g_instr_sum_death_prob, sizeof(double));
    return host;
}

void resetInstrCounters() {
    unsigned long long zero = 0;
    double zero_d = 0.0;
    cudaMemcpyToSymbol(g_instr_repair_decisions, &zero, sizeof(zero));
    cudaMemcpyToSymbol(g_instr_repair_kills, &zero, sizeof(zero));
    cudaMemcpyToSymbol(g_instr_irradiation_events, &zero, sizeof(zero));
    cudaMemcpyToSymbol(g_instr_divisions, &zero, sizeof(zero));
    cudaMemcpyToSymbol(g_instr_sum_death_prob, &zero_d, sizeof(zero_d));
    std::vector<unsigned long long> zeros(INSTR_HIST_N, 0);
    cudaMemcpyToSymbol(g_instr_irr_hist, zeros.data(), sizeof(unsigned long long) * INSTR_HIST_N);
    cudaMemcpyToSymbol(g_instr_tir_hist, zeros.data(), sizeof(unsigned long long) * INSTR_HIST_N);
}

std::vector<unsigned long long> readInstrHistograms() {
    std::vector<unsigned long long> out(2 * INSTR_HIST_N, 0);
    cudaMemcpyFromSymbol(out.data(), g_instr_irr_hist,
                         sizeof(unsigned long long) * INSTR_HIST_N);
    cudaMemcpyFromSymbol(out.data() + INSTR_HIST_N, g_instr_tir_hist,
                         sizeof(unsigned long long) * INSTR_HIST_N);
    return out;
}

}  // namespace emt6ro
