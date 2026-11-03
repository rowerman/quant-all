#include "quant_native/binning.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace quant_native {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

template <typename Function>
void parallel_for(std::size_t count, int threads, Function function) {
    const std::size_t worker_count = std::min<std::size_t>(
        static_cast<std::size_t>(threads),
        std::max<std::size_t>(count, 1)
    );
    if (worker_count == 1) {
        function(0, 1);
        return;
    }

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> failures(worker_count);
    workers.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
        workers.emplace_back([&, index]() {
            try {
                function(index, worker_count);
            } catch (...) {
                failures[index] = std::current_exception();
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (const std::exception_ptr& failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
}

std::vector<double> average_ranks(
    const std::vector<double>& values,
    const std::vector<std::size_t>& rows
) {
    std::vector<std::size_t> order = rows;
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](std::size_t left, std::size_t right) {
            return values[left] < values[right];
        }
    );
    std::vector<double> ranks(values.size(), 0.0);
    std::size_t begin = 0;
    while (begin < order.size()) {
        std::size_t finish = begin + 1;
        const double tied_value = values[order[begin]];
        while (
            finish < order.size()
            && values[order[finish]] == tied_value
        ) {
            ++finish;
        }
        const double average =
            (static_cast<double>(begin) + static_cast<double>(finish) + 1.0)
            / 2.0;
        for (std::size_t position = begin; position < finish; ++position) {
            ranks[order[position]] = average;
        }
        begin = finish;
    }
    return ranks;
}

void row_correlation(
    const double* left,
    const double* right,
    std::size_t columns,
    std::size_t row,
    int min_n,
    double* correlation,
    std::int64_t* count
) {
    const std::size_t start = row * columns;
    std::vector<double> left_values;
    std::vector<double> right_values;
    std::vector<std::size_t> positions;
    left_values.reserve(columns);
    right_values.reserve(columns);
    positions.reserve(columns);
    for (std::size_t column = 0; column < columns; ++column) {
        const double left_value = left[start + column];
        const double right_value = right[start + column];
        if (std::isfinite(left_value) && std::isfinite(right_value)) {
            positions.push_back(positions.size());
            left_values.push_back(left_value);
            right_values.push_back(right_value);
        }
    }
    count[row] = static_cast<std::int64_t>(positions.size());
    correlation[row] = kNaN;
    if (positions.size() < static_cast<std::size_t>(min_n)) {
        return;
    }
    std::vector<std::size_t> order(positions.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
    }
    std::vector<double> left_ranks = average_ranks(left_values, order);
    std::vector<double> right_ranks = average_ranks(right_values, order);
    const double center =
        (static_cast<double>(positions.size()) + 1.0) / 2.0;
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    for (std::size_t index = 0; index < positions.size(); ++index) {
        const double left_centered = left_ranks[index] - center;
        const double right_centered = right_ranks[index] - center;
        dot += left_centered * right_centered;
        left_norm += left_centered * left_centered;
        right_norm += right_centered * right_centered;
    }
    const double scale = std::sqrt(left_norm * right_norm);
    if (scale > 0.0 && std::isfinite(scale)) {
        correlation[row] = std::clamp(dot / scale, -1.0, 1.0);
    }
}

}  // namespace

void bin_turnover(
    const double* values,
    std::size_t rows,
    std::size_t columns,
    int quantiles,
    int threads,
    double* per_bin,
    double* overall
) {
    if (
        rows == 0 || columns == 0 || quantiles < 1 || threads < 1
    ) {
        throw std::invalid_argument("bin_turnover 输入或参数非法");
    }
    std::fill(
        per_bin,
        per_bin + rows * static_cast<std::size_t>(quantiles),
        kNaN
    );
    std::fill(overall, overall + rows, kNaN);

    parallel_for(rows, threads, [&](std::size_t worker, std::size_t count) {
        for (std::size_t row = worker; row < rows; row += count) {
            if (row == 0) {
                continue;
            }
            std::size_t base = 0;
            std::size_t changed = 0;
            std::vector<std::size_t> bin_base(
                static_cast<std::size_t>(quantiles),
                0
            );
            std::vector<std::size_t> bin_changed(
                static_cast<std::size_t>(quantiles),
                0
            );
            for (std::size_t column = 0; column < columns; ++column) {
                const double current = values[row * columns + column];
                const double previous =
                    values[(row - 1) * columns + column];
                if (std::isnan(current) || std::isnan(previous)) {
                    continue;
                }
                ++base;
                if (current != previous) {
                    ++changed;
                }
                if (
                    current < 0.0
                    || current >= static_cast<double>(quantiles)
                    || current != std::floor(current)
                ) {
                    continue;
                }
                const auto bin = static_cast<std::size_t>(current);
                ++bin_base[bin];
                if (previous != current) {
                    ++bin_changed[bin];
                }
            }
            if (base > 0) {
                overall[row] =
                    static_cast<double>(changed) / static_cast<double>(base);
            }
            for (std::size_t bin = 0; bin < bin_base.size(); ++bin) {
                if (bin_base[bin] > 0) {
                    per_bin[row * bin_base.size() + bin] =
                        static_cast<double>(bin_changed[bin])
                        / static_cast<double>(bin_base[bin]);
                }
            }
        }
    });
}

void daily_row_spearman(
    const double* left,
    const double* right,
    std::size_t rows,
    std::size_t columns,
    int min_n,
    int threads,
    double* correlation,
    std::int64_t* count
) {
    if (
        rows == 0 || columns == 0 || min_n < 2 || threads < 1
    ) {
        throw std::invalid_argument("daily_row_spearman 输入或参数非法");
    }
    parallel_for(rows, threads, [&](std::size_t worker, std::size_t count_threads) {
        for (std::size_t row = worker; row < rows; row += count_threads) {
            row_correlation(
                left,
                right,
                columns,
                row,
                min_n,
                correlation,
                count
            );
        }
    });
}

}  // namespace quant_native
