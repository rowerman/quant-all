#pragma once

#include <cstddef>
#include <cstdint>

namespace quant_native {

void pairwise_spearman(
    const double* values,
    std::size_t rows,
    std::size_t columns,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int min_n,
    int threads,
    double* correlation,
    std::int64_t* count
);

int threads_from_environment();

}  // namespace quant_native
