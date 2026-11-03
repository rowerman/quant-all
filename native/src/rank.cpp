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

void rank_group(
    const double* values,
    std::size_t start,
    std::size_t end,
    double* result
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
        const double average =
            (static_cast<double>(begin) + static_cast<double>(finish) + 1.0)
            / 2.0;
        const double percentile = average / sample_count;
        for (std::size_t position = begin; position < finish; ++position) {
            result[order[position]] = percentile;
        }
        begin = finish;
    }
}

}  // namespace

void daily_rank_pct(
    const double* values,
    std::size_t rows,
    const std::int64_t* group_offsets,
    std::size_t group_count,
    int threads,
    double* result
) {
    if (rows == 0 || group_count == 0 || threads < 1) {
        throw std::invalid_argument("daily_rank_pct 输入不能为空");
    }
    std::fill(
        result,
        result + rows,
        std::numeric_limits<double>::quiet_NaN()
    );

    auto worker = [&](std::size_t worker_index, std::size_t worker_count) {
        for (std::size_t group = worker_index;
             group < group_count;
             group += worker_count) {
            rank_group(
                values,
                static_cast<std::size_t>(group_offsets[group]),
                static_cast<std::size_t>(group_offsets[group + 1]),
                result
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
