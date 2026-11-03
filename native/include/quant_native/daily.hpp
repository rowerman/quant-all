#pragma once

#include <cstddef>
#include <cstdint>

namespace quant_native {

void daily_rank_pct(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int threads,
    double* result
);

void daily_pearson(
    const double* left,
    const double* right,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int min_n,
    int threads,
    double* correlation,
    std::int64_t* count
);

void daily_winsorize_mad(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    double n_mad,
    int threads,
    double* result
);

void daily_zscore(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int threads,
    double* result
);

void daily_zscore_population(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int threads,
    double* result
);

void daily_winsorize_quantile(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    double lower_quantile,
    double upper_quantile,
    int threads,
    double* result
);

void daily_winsorize_zscore(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    double n_mad,
    int threads,
    double* result
);

void daily_winsorize_zscore_columns(
    const double* values,
    std::size_t rows,
    std::size_t columns,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    double n_mad,
    int threads,
    double* result
);

}  // namespace quant_native
