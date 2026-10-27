#include "quant_native/spearman.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

namespace quant_native {
namespace {

using Mask = std::vector<std::uint64_t>;

struct RankedColumn {
    std::vector<double> values;
    double norm = 0.0;
};

struct MaskWorkspace {
    std::size_t start = 0;
    std::size_t end = 0;
    std::vector<std::size_t> valid_rows;
    std::vector<int> local_positions;
    std::map<std::size_t, RankedColumn> ranks;
};

Mask column_mask(
    const double* values,
    std::size_t columns,
    std::size_t column,
    std::size_t start,
    std::size_t end
) {
    const std::size_t group_rows = end - start;
    Mask mask((group_rows + 63) / 64, 0);
    for (std::size_t row = start; row < end; ++row) {
        if (std::isfinite(values[row * columns + column])) {
            const std::size_t local = row - start;
            mask[local / 64] |= std::uint64_t{1} << (local % 64);
        }
    }
    return mask;
}

Mask intersection(const Mask& left, const Mask& right) {
    Mask result(left.size(), 0);
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] & right[index];
    }
    return result;
}

std::size_t mask_count(const Mask& mask) {
    std::size_t result = 0;
    for (const std::uint64_t word : mask) {
        result += std::popcount(word);
    }
    return result;
}

MaskWorkspace make_workspace(
    const Mask& mask,
    std::size_t group_rows,
    std::size_t start,
    std::size_t end
) {
    MaskWorkspace workspace;
    workspace.start = start;
    workspace.end = end;
    workspace.local_positions.assign(group_rows, -1);
    workspace.valid_rows.reserve(mask_count(mask));
    for (std::size_t local = 0; local < group_rows; ++local) {
        if ((mask[local / 64] & (std::uint64_t{1} << (local % 64))) == 0) {
            continue;
        }
        workspace.local_positions[local] =
            static_cast<int>(workspace.valid_rows.size());
        workspace.valid_rows.push_back(start + local);
    }
    return workspace;
}

RankedColumn rank_column(
    const double* values,
    std::size_t columns,
    std::size_t column,
    const MaskWorkspace& workspace,
    const std::vector<std::size_t>& order
) {
    const std::size_t sample_count = workspace.valid_rows.size();
    std::vector<std::size_t> ordered_rows;
    ordered_rows.reserve(sample_count);
    for (const std::size_t row : order) {
        if (row < workspace.start || row >= workspace.end) {
            continue;
        }
        if (workspace.local_positions[row - workspace.start] >= 0) {
            ordered_rows.push_back(row);
        }
    }

    std::vector<double> ranks(sample_count);
    std::size_t begin = 0;
    while (begin < sample_count) {
        std::size_t finish = begin + 1;
        const double tied_value =
            values[ordered_rows[begin] * columns + column];
        while (finish < sample_count) {
            const double value =
                values[ordered_rows[finish] * columns + column];
            if (value != tied_value) {
                break;
            }
            ++finish;
        }
        const double average =
            (static_cast<double>(begin) + static_cast<double>(finish) + 1.0) / 2.0;
        for (std::size_t position = begin; position < finish; ++position) {
            const std::size_t row = ordered_rows[position];
            const int local = workspace.local_positions[row - workspace.start];
            ranks[static_cast<std::size_t>(local)] = average;
        }
        begin = finish;
    }

    const double mean = (static_cast<double>(sample_count) + 1.0) / 2.0;
    double squared_sum = 0.0;
    for (double& rank : ranks) {
        rank -= mean;
        squared_sum += rank * rank;
    }
    const double norm = std::sqrt(squared_sum);
    if (!(norm > 0.0) || !std::isfinite(norm)) {
        return {std::move(ranks), norm};
    }
    for (std::size_t position = 0; position < sample_count; ++position) {
        ranks[position] /= norm;
    }
    return {std::move(ranks), norm};
}

void compute_group(
    const double* values,
    std::size_t columns,
    std::size_t start,
    std::size_t end,
    int min_n,
    double* group_correlation,
    std::int64_t* group_count,
    const std::vector<std::vector<std::size_t>>& column_orders
) {
    const std::size_t group_rows = end - start;
    std::vector<Mask> masks;
    masks.reserve(columns);
    for (std::size_t column = 0; column < columns; ++column) {
        masks.push_back(
            column_mask(values, columns, column, start, end)
        );
    }

    for (std::size_t left = 0; left < columns; ++left) {
        for (std::size_t right = left; right < columns; ++right) {
            const auto sample_count = static_cast<std::int64_t>(
                mask_count(intersection(masks[left], masks[right]))
            );
            group_count[left * columns + right] = sample_count;
            group_count[right * columns + left] = sample_count;
        }
    }

    std::map<Mask, std::vector<std::size_t>> mask_groups;
    for (std::size_t column = 0; column < columns; ++column) {
        mask_groups[masks[column]].push_back(column);
    }
    std::vector<std::vector<std::size_t>> groups;
    groups.reserve(mask_groups.size());
    for (auto& entry : mask_groups) {
        groups.push_back(std::move(entry.second));
    }

    std::map<Mask, MaskWorkspace> workspaces;
    for (std::size_t left_group = 0; left_group < groups.size(); ++left_group) {
        for (std::size_t right_group = left_group;
             right_group < groups.size();
             ++right_group) {
            const auto& left_columns = groups[left_group];
            const auto& right_columns = groups[right_group];
            Mask key = intersection(
                masks[left_columns[0]],
                masks[right_columns[0]]
            );
            if (static_cast<int>(mask_count(key)) < min_n) {
                continue;
            }
            auto workspace_entry = workspaces.find(key);
            if (workspace_entry == workspaces.end()) {
                workspace_entry = workspaces.emplace(
                    key,
                    make_workspace(key, group_rows, start, end)
                ).first;
            }
            MaskWorkspace& workspace = workspace_entry->second;

            std::vector<const RankedColumn*> left_ranked;
            std::vector<const RankedColumn*> right_ranked;
            left_ranked.reserve(left_columns.size());
            if (left_group != right_group) {
                right_ranked.reserve(right_columns.size());
            }
            auto ranked = [&](std::size_t column) -> const RankedColumn& {
                auto found = workspace.ranks.find(column);
                if (found == workspace.ranks.end()) {
                    found = workspace.ranks.emplace(
                        column,
                        rank_column(
                            values,
                            columns,
                            column,
                            workspace,
                            column_orders[column]
                        )
                    ).first;
                }
                return found->second;
            };
            for (const std::size_t column : left_columns) {
                left_ranked.push_back(&ranked(column));
            }
            if (left_group != right_group) {
                for (const std::size_t column : right_columns) {
                    right_ranked.push_back(&ranked(column));
                }
            } else {
                right_ranked = left_ranked;
            }

            for (std::size_t left_local = 0;
                 left_local < left_columns.size();
                 ++left_local) {
                const RankedColumn& left_values = *left_ranked[left_local];
                if (!(left_values.norm > 0.0)) {
                    continue;
                }
                const std::size_t right_start =
                    left_group == right_group ? left_local : 0;
                for (std::size_t right_local = right_start;
                     right_local < right_columns.size();
                     ++right_local) {
                    const RankedColumn& right_values = *right_ranked[right_local];
                    if (!(right_values.norm > 0.0)) {
                        continue;
                    }
                    double value = 0.0;
                    for (std::size_t position = 0;
                         position < left_values.values.size();
                         ++position) {
                        value +=
                            left_values.values[position]
                            * right_values.values[position];
                    }
                    value = std::clamp(value, -1.0, 1.0);
                    const std::size_t left_column = left_columns[left_local];
                    const std::size_t right_column = right_columns[right_local];
                    group_correlation[left_column * columns + right_column] = value;
                    group_correlation[right_column * columns + left_column] = value;
                }
            }
        }
    }
}

}  // namespace

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
) {
    if (rows == 0 || columns == 0 || group_count == 0 || threads < 1) {
        throw std::invalid_argument("pairwise_spearman 输入不能为空");
    }
    const std::size_t matrix_size = columns * columns;
    std::fill(
        correlation,
        correlation + group_count * matrix_size,
        std::numeric_limits<double>::quiet_NaN()
    );
    std::fill(
        count,
        count + group_count * matrix_size,
        static_cast<std::int64_t>(0)
    );
    std::vector<std::vector<std::size_t>> column_orders(columns);
    for (std::size_t column = 0; column < columns; ++column) {
        auto& order = column_orders[column];
        order.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            if (std::isfinite(values[row * columns + column])) {
                order.push_back(row);
            }
        }
        std::stable_sort(
            order.begin(),
            order.end(),
            [&](std::size_t left, std::size_t right) {
                return values[left * columns + column]
                    < values[right * columns + column];
            }
        );
    }

    auto worker = [&](std::size_t worker_index, std::size_t worker_count) {
        for (std::size_t group = worker_index;
             group < group_count;
             group += worker_count) {
            const auto start = static_cast<std::size_t>(group_offsets[group]);
            const auto end = static_cast<std::size_t>(group_offsets[group + 1]);
            double* group_correlation = correlation + group * matrix_size;
            std::int64_t* group_count_values = count + group * matrix_size;
            compute_group(
                values,
                columns,
                start,
                end,
                min_n,
                group_correlation,
                group_count_values,
                column_orders
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

int threads_from_environment() {
    const char* raw = std::getenv("QUANT_NATIVE_THREADS");
    if (raw == nullptr) {
        return 1;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (
        errno != 0
        || end == raw
        || *end != '\0'
        || parsed < 1
        || parsed > std::numeric_limits<int>::max()
    ) {
        throw std::invalid_argument("QUANT_NATIVE_THREADS 必须是正整数");
    }
    return static_cast<int>(parsed);
}

}  // namespace quant_native
