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

void quantile_group(
    const double* values,
    std::size_t start,
    std::size_t end,
    int quantiles,
    std::int64_t* result
) {
    std::vector<std::size_t> order;
    order.reserve(end - start);
    for (std::size_t row = start; row < end; ++row) {
        if (!std::isnan(values[row])) {
            order.push_back(row);
        }
    }
    if (order.empty()) {
        return;
    }
    std::stable_sort(
        order.begin(),
        order.end(),
        [&](std::size_t left, std::size_t right) {
            return values[left] < values[right];
        }
    );

    const double sample_count = static_cast<double>(order.size());
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
        const double average_rank =
            (static_cast<double>(begin) + static_cast<double>(finish) + 1.0)
            / 2.0;
        const double raw_bin =
            std::ceil(average_rank / sample_count * quantiles);
        const auto bin = static_cast<std::int64_t>(
            std::clamp(raw_bin, 1.0, static_cast<double>(quantiles))
        );
        for (std::size_t position = begin; position < finish; ++position) {
            result[order[position]] = bin - 1;
        }
        begin = finish;
    }
}

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

}  // namespace

void daily_quantile_bins(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int quantiles,
    int threads,
    std::int64_t* result
) {
    if (rows == 0 || group_count == 0 || quantiles < 2 || threads < 1) {
        throw std::invalid_argument("daily_quantile_bins 输入或参数非法");
    }
    std::fill(result, result + rows, std::int64_t{-1});
    parallel_for(group_count, threads, [&](std::size_t worker, std::size_t count) {
        for (std::size_t group = worker; group < group_count; group += count) {
            quantile_group(
                values,
                static_cast<std::size_t>(group_offsets[group]),
                static_cast<std::size_t>(group_offsets[group + 1]),
                quantiles,
                result
            );
        }
    });
}

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
) {
    if (
        rows == 0 || labels == 0 || day_count == 0 || bin_count == 0
        || threads < 1
    ) {
        throw std::invalid_argument("bin_returns_multi 输入或参数非法");
    }
    const std::size_t return_count = day_count * bin_count * labels;
    const std::size_t pool_count = day_count * labels;
    std::fill(returns, returns + return_count, kNaN);
    std::fill(pools, pools + pool_count, kNaN);

    std::vector<double> numerators(return_count, 0.0);
    std::vector<double> denominators(return_count, 0.0);
    std::vector<double> pool_numerators(pool_count, 0.0);
    std::vector<double> pool_denominators(pool_count, 0.0);

    for (std::size_t row = 0; row < rows; ++row) {
        const double weight = weights[row];
        const std::int64_t day = day_codes[row];
        const std::int64_t bin = bin_codes[row];
        if (
            !std::isfinite(weight) || weight <= 0.0
            || day < 0 || static_cast<std::size_t>(day) >= day_count
            || bin < 0 || static_cast<std::size_t>(bin) >= bin_count
        ) {
            continue;
        }
        for (std::size_t label = 0; label < labels; ++label) {
            const double value = values[row * labels + label];
            if (!std::isfinite(value)) {
                continue;
            }
            const std::size_t flat =
                static_cast<std::size_t>(day) * bin_count * labels
                + static_cast<std::size_t>(bin) * labels
                + label;
            const double weighted = value * weight;
            numerators[flat] += weighted;
            denominators[flat] += weight;
            pool_numerators[static_cast<std::size_t>(day) * labels + label]
                += weighted;
            pool_denominators[
                static_cast<std::size_t>(day) * labels + label
            ] += weight;
        }
    }

    parallel_for(day_count, threads, [&](std::size_t worker, std::size_t count) {
        for (std::size_t day = worker; day < day_count; day += count) {
            for (std::size_t bin = 0; bin < bin_count; ++bin) {
                for (std::size_t label = 0; label < labels; ++label) {
                    const std::size_t flat =
                        day * bin_count * labels + bin * labels + label;
                    if (denominators[flat] > 0.0) {
                        returns[flat] =
                            numerators[flat] / denominators[flat];
                    }
                }
            }
            for (std::size_t label = 0; label < labels; ++label) {
                const std::size_t flat = day * labels + label;
                if (pool_denominators[flat] > 0.0) {
                    pools[flat] =
                        pool_numerators[flat] / pool_denominators[flat];
                }
            }
        }
    });
}

}  // namespace quant_native
