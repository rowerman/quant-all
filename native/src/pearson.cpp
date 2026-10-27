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

void pearson_group(
    const double* left,
    const double* right,
    std::size_t start,
    std::size_t end,
    int min_n,
    double* correlation,
    std::int64_t* count
) {
    std::size_t sample_count = 0;
    double sum_left = 0.0;
    double sum_right = 0.0;
    for (std::size_t row = start; row < end; ++row) {
        if (std::isfinite(left[row]) && std::isfinite(right[row])) {
            sum_left += left[row];
            sum_right += right[row];
            ++sample_count;
        }
    }
    *count = static_cast<std::int64_t>(sample_count);
    *correlation = std::numeric_limits<double>::quiet_NaN();
    if (
        sample_count < static_cast<std::size_t>(min_n)
        || sample_count == 0
    ) {
        return;
    }

    const double mean_left = sum_left / static_cast<double>(sample_count);
    const double mean_right = sum_right / static_cast<double>(sample_count);
    double correction_left = 0.0;
    double correction_right = 0.0;
    for (std::size_t row = start; row < end; ++row) {
        if (std::isfinite(left[row]) && std::isfinite(right[row])) {
            correction_left += left[row] - mean_left;
            correction_right += right[row] - mean_right;
        }
    }
    correction_left /= static_cast<double>(sample_count);
    correction_right /= static_cast<double>(sample_count);

    double covariance = 0.0;
    double variance_left = 0.0;
    double variance_right = 0.0;
    for (std::size_t row = start; row < end; ++row) {
        if (!std::isfinite(left[row]) || !std::isfinite(right[row])) {
            continue;
        }
        const double left_deviation =
            left[row] - mean_left - correction_left;
        const double right_deviation =
            right[row] - mean_right - correction_right;
        covariance += left_deviation * right_deviation;
        variance_left += left_deviation * left_deviation;
        variance_right += right_deviation * right_deviation;
    }
    const double scale = std::sqrt(variance_left * variance_right);
    if (scale > 0.0 && std::isfinite(scale)) {
        *correlation = std::clamp(covariance / scale, -1.0, 1.0);
    }
}

}  // namespace

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
) {
    if (rows == 0 || group_count == 0 || threads < 1 || min_n < 2) {
        throw std::invalid_argument("daily_pearson 输入或参数非法");
    }

    auto worker = [&](std::size_t worker_index, std::size_t worker_count) {
        for (std::size_t group = worker_index;
             group < group_count;
             group += worker_count) {
            pearson_group(
                left,
                right,
                static_cast<std::size_t>(group_offsets[group]),
                static_cast<std::size_t>(group_offsets[group + 1]),
                min_n,
                correlation + group,
                count + group
            );
        }
    };

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

}  // namespace quant_native
