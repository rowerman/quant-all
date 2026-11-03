#pragma once

#include <cstddef>
#include <cstdint>

namespace quant_native {

void daily_ols(
    const double* design,
    const double* response,
    std::size_t rows,
    std::size_t columns,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int min_rows,
    int threads,
    double* coefficients,
    double* t_stats,
    double* r2,
    std::int64_t* sample_rows,
    std::uint8_t* active,
    std::uint8_t* group_used,
    std::uint8_t* group_fallback,
    double* residual_values,
    std::uint8_t* residual_used,
    std::int64_t* skip_counts
);

}  // namespace quant_native
