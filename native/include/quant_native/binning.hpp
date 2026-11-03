#pragma once

#include <cstddef>
#include <cstdint>

namespace quant_native {

void daily_quantile_bins(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int quantiles,
    int threads,
    std::int64_t* result
);

void bin_returns_multi(
    const double* values,
    std::size_t rows,
    std::size_t labels,
    const std::int64_t* day_codes,
    const std::int64_t* bin_codes,
    const double* weights,
    std::size_t day_count,
    std::size_t bin_count,
    int threads,
    double* returns,
    double* pools
);

void bin_turnover(
    const double* values,
    std::size_t rows,
    std::size_t columns,
    int quantiles,
    int threads,
    double* per_bin,
    double* overall
);

void daily_row_spearman(
    const double* left,
    const double* right,
    std::size_t rows,
    std::size_t columns,
    int min_n,
    int threads,
    double* correlation,
    std::int64_t* count
);

}  // namespace quant_native
