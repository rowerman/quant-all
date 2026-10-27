#include "quant_native/daily.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace quant_native {
namespace {

constexpr double kMadScale = 1.4826;

double median(std::vector<double>& values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 != 0) {
        return upper;
    }
    const auto lower = std::max_element(values.begin(), values.begin() + middle);
    return (*lower + upper) / 2.0;
}

std::vector<double> winsorize_group(
    const double* values,
    std::size_t start,
    std::size_t end,
    double n_mad
) {
    std::vector<double> result(values + start, values + end);
    std::vector<double> samples;
    samples.reserve(result.size());
    for (double value : result) {
        if (!std::isnan(value)) {
            samples.push_back(value);
        }
    }
    const double center = median(samples);
    if (std::isnan(center)) {
        return result;
    }

    samples.clear();
    samples.reserve(result.size());
    for (double value : result) {
        if (std::isnan(value)) {
            continue;
        }
        const double deviation = std::abs(value - center);
        if (!std::isnan(deviation)) {
            samples.push_back(deviation);
        }
    }
    const double mad = median(samples);
    if (!(mad > 0.0)) {
        return result;
    }

    const double limit = n_mad * kMadScale * mad;
    const double lower = center - limit;
    const double upper = center + limit;
    if (std::isnan(lower) || std::isnan(upper)) {
        std::fill(
            result.begin(),
            result.end(),
            std::numeric_limits<double>::quiet_NaN()
        );
        return result;
    }
    for (double& value : result) {
        if (std::isnan(value)) {
            continue;
        }
        if (value < lower) {
            value = lower;
        } else if (value > upper) {
            value = upper;
        }
    }
    return result;
}

void zscore_group(
    const double* values,
    std::size_t start,
    std::size_t end,
    double* result
) {
    std::size_t count = 0;
    double sum = 0.0;
    double sum_compensation = 0.0;
    double square_sum = 0.0;
    double square_compensation = 0.0;
    for (std::size_t row = start; row < end; ++row) {
        if (std::isfinite(values[row])) {
            const double corrected = values[row] - sum_compensation;
            const double updated = sum + corrected;
            sum_compensation = (updated - sum) - corrected;
            sum = updated;

            const double square = values[row] * values[row];
            const double corrected_square = square - square_compensation;
            const double updated_square = square_sum + corrected_square;
            square_compensation =
                (updated_square - square_sum) - corrected_square;
            square_sum = updated_square;
            ++count;
        }
    }
    if (count < 2) {
        std::fill(
            result + start,
            result + end,
            std::numeric_limits<double>::quiet_NaN()
        );
        return;
    }

    const double sample_count = static_cast<double>(count);
    const double mean = sum / sample_count;
    double variance =
        (square_sum - sample_count * (mean * mean)) / (sample_count - 1.0);
    variance = std::max(variance, 0.0);
    const double standard_deviation = std::sqrt(variance);
    if (!(standard_deviation > 0.0)) {
        std::fill(
            result + start,
            result + end,
            std::numeric_limits<double>::quiet_NaN()
        );
        return;
    }
    for (std::size_t row = start; row < end; ++row) {
        result[row] = (values[row] - mean) / standard_deviation;
    }
}

template <typename Worker>
void run_groups(
    std::size_t group_count,
    int threads,
    Worker worker
) {
    const std::size_t worker_count = std::min<std::size_t>(
        static_cast<std::size_t>(threads),
        std::max<std::size_t>(group_count, 1)
    );
    if (worker_count == 1) {
        worker(0, 1);
        return;
    }

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> failures(worker_count);
    workers.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
        workers.emplace_back([&, index]() {
            try {
                worker(index, worker_count);
            } catch (...) {
                failures[index] = std::current_exception();
            }
        });
    }
    for (std::thread& thread : workers) {
        thread.join();
    }
    for (const std::exception_ptr& failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
}

void validate_inputs(
    std::size_t rows,
    std::size_t group_count,
    int threads,
    double n_mad
) {
    if (
        rows == 0
        || group_count == 0
        || threads < 1
        || !std::isfinite(n_mad)
        || n_mad <= 0.0
    ) {
        throw std::invalid_argument("逐日标准化输入或参数非法");
    }
}

}  // namespace

void daily_winsorize_mad(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    double n_mad,
    int threads,
    double* result
) {
    validate_inputs(rows, group_count, threads, n_mad);
    run_groups(
        group_count,
        threads,
        [&](std::size_t worker_index, std::size_t worker_count) {
            for (std::size_t group = worker_index;
                 group < group_count;
                 group += worker_count) {
                const std::size_t start =
                    static_cast<std::size_t>(group_offsets[group]);
                const std::size_t end =
                    static_cast<std::size_t>(group_offsets[group + 1]);
                const std::vector<double> clipped =
                    winsorize_group(values, start, end, n_mad);
                std::copy(clipped.begin(), clipped.end(), result + start);
            }
        }
    );
}

void daily_zscore(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int threads,
    double* result
) {
    validate_inputs(rows, group_count, threads, 1.0);
    run_groups(
        group_count,
        threads,
        [&](std::size_t worker_index, std::size_t worker_count) {
            for (std::size_t group = worker_index;
                 group < group_count;
                 group += worker_count) {
                zscore_group(
                    values,
                    static_cast<std::size_t>(group_offsets[group]),
                    static_cast<std::size_t>(group_offsets[group + 1]),
                    result
                );
            }
        }
    );
}

void daily_winsorize_zscore(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    double n_mad,
    int threads,
    double* result
) {
    validate_inputs(rows, group_count, threads, n_mad);
    run_groups(
        group_count,
        threads,
        [&](std::size_t worker_index, std::size_t worker_count) {
            for (std::size_t group = worker_index;
                 group < group_count;
                 group += worker_count) {
                const std::size_t start =
                    static_cast<std::size_t>(group_offsets[group]);
                const std::size_t end =
                    static_cast<std::size_t>(group_offsets[group + 1]);
                const std::vector<double> clipped =
                    winsorize_group(values, start, end, n_mad);
                zscore_group(
                    clipped.data(),
                    0,
                    clipped.size(),
                    result + start
                );
            }
        }
    );
}

}  // namespace quant_native
