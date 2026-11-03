#include "quant_native/ols.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace quant_native {
namespace {

constexpr double kActiveRange = 1e-12;
constexpr double kZeroResidualRatio = 1e-12;
constexpr double kFallbackCondition = 1e3;
constexpr int kInsufficientRows = 0;
constexpr int kInvalidResponse = 1;
constexpr int kRankDeficient = 2;

double column_range(
    const double* design,
    std::size_t columns,
    std::size_t start,
    std::size_t end,
    std::size_t column
) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t row = start; row < end; ++row) {
        const double value = design[row * columns + column];
        if (!std::isfinite(value)) {
            throw std::runtime_error("daily_ols 设计矩阵包含非有限值");
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    return maximum - minimum;
}

void run_group(
    const double* design,
    const double* response,
    std::size_t columns,
    std::size_t start,
    std::size_t end,
    int min_rows,
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
) {
    const std::size_t size = end - start;
    if (size < static_cast<std::size_t>(min_rows)) {
        if (size > 0) {
            skip_counts[kInsufficientRows] = 1;
        }
        return;
    }

    double response_min = std::numeric_limits<double>::infinity();
    double response_max = -std::numeric_limits<double>::infinity();
    for (std::size_t row = start; row < end; ++row) {
        const double value = response[row];
        if (!std::isfinite(value)) {
            skip_counts[kInvalidResponse] = 1;
            return;
        }
        response_min = std::min(response_min, value);
        response_max = std::max(response_max, value);
    }
    if (response_max == response_min) {
        skip_counts[kInvalidResponse] = 1;
        return;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t column = 0; column < columns; ++column) {
        active[column] =
            column == 0
            || column_range(design, columns, start, end, column) > kActiveRange;
        coefficients[column] = nan;
        t_stats[column] = nan;
    }
    const std::size_t active_columns = static_cast<std::size_t>(
        std::count(active, active + columns, 1)
    );
    Eigen::MatrixXd reduced(size, static_cast<Eigen::Index>(active_columns));
    std::size_t active_index = 0;
    for (std::size_t column = 0; column < columns; ++column) {
        if (!active[column]) {
            continue;
        }
        for (std::size_t row = 0; row < size; ++row) {
            reduced(
                static_cast<Eigen::Index>(row),
                static_cast<Eigen::Index>(active_index)
            ) = design[(start + row) * columns + column];
        }
        ++active_index;
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(reduced);
    qr.setThreshold(
        std::numeric_limits<double>::epsilon()
        * static_cast<double>(std::max(reduced.rows(), reduced.cols()))
    );
    const Eigen::Index active_size =
        static_cast<Eigen::Index>(active_columns);
    const Eigen::MatrixXd triangular = qr.matrixR();
    const double largest_pivot =
        active_size > 0 ? std::abs(triangular(0, 0)) : 0.0;
    const double smallest_pivot =
        active_size > 0
        ? std::abs(triangular(active_size - 1, active_size - 1))
        : 0.0;
    if (
        smallest_pivot <= 0.0
        || largest_pivot / smallest_pivot > kFallbackCondition
    ) {
        *group_fallback = 1;
        return;
    }
    const std::size_t rank = static_cast<std::size_t>(qr.rank());
    if (rank < active_columns || size <= rank) {
        skip_counts[kRankDeficient] = 1;
        return;
    }

    Eigen::Map<const Eigen::VectorXd> response_day(response + start, size);
    const Eigen::VectorXd beta_active = qr.solve(response_day);
    const Eigen::VectorXd fitted = reduced * beta_active;
    Eigen::VectorXd output_residual = response_day - fitted;
    if (
        output_residual.norm()
        <= kZeroResidualRatio * response_day.norm()
    ) {
        output_residual.setZero();
    }

    const double degrees = static_cast<double>(size - rank);
    const double sigma2 = output_residual.squaredNorm() / degrees;
    Eigen::JacobiSVD<Eigen::MatrixXd> covariance_svd(
        reduced,
        Eigen::ComputeThinV
    );
    const Eigen::VectorXd singular_values =
        covariance_svd.singularValues();
    const Eigen::VectorXd inverse_square =
        singular_values.cwiseProduct(singular_values).cwiseInverse();
    const Eigen::MatrixXd covariance =
        sigma2
        * covariance_svd.matrixV()
        * inverse_square.asDiagonal()
        * covariance_svd.matrixV().transpose();
    active_index = 0;
    for (std::size_t column = 0; column < columns; ++column) {
        if (!active[column]) {
            continue;
        }
        const Eigen::Index index = static_cast<Eigen::Index>(active_index);
        const double beta = beta_active(index);
        const double standard_error = std::sqrt(
            std::max(covariance(index, index), 0.0)
        );
        coefficients[column] = beta;
        t_stats[column] = beta / standard_error;
        ++active_index;
    }
    const double response_mean = response_day.mean();
    const double total =
        response_day.squaredNorm()
        - static_cast<double>(size) * response_mean * response_mean;
    *r2 =
        total > 0.0
        ? 1.0 - output_residual.squaredNorm() / total
        : nan;
    *sample_rows = static_cast<std::int64_t>(size);
    *group_used = 1;
    for (std::size_t row = 0; row < size; ++row) {
        residual_values[start + row] =
            output_residual(static_cast<Eigen::Index>(row));
        residual_used[start + row] = 1;
    }
}

}  // namespace

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
) {
    if (
        rows == 0
        || columns == 0
        || group_count == 0
        || min_rows < 1
        || threads < 1
    ) {
        throw std::invalid_argument("daily_ols 输入或参数非法");
    }
    std::vector<std::int64_t> group_skip_counts(group_count * 3, 0);
    auto worker = [&](std::size_t worker_index, std::size_t worker_count) {
        for (std::size_t group = worker_index;
             group < group_count;
             group += worker_count) {
            run_group(
                design,
                response,
                columns,
                static_cast<std::size_t>(group_offsets[group]),
                static_cast<std::size_t>(group_offsets[group + 1]),
                min_rows,
                coefficients + group * columns,
                t_stats + group * columns,
                r2 + group,
                sample_rows + group,
                active + group * columns,
                group_used + group,
                group_fallback + group,
                residual_values,
                residual_used,
                group_skip_counts.data() + group * 3
            );
        }
    };

    const std::size_t worker_count = std::min<std::size_t>(
        static_cast<std::size_t>(threads),
        group_count
    );
    if (worker_count == 1) {
        worker(0, 1);
    } else {
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
    std::fill(skip_counts, skip_counts + 3, 0);
    for (std::size_t group = 0; group < group_count; ++group) {
        for (int reason = 0; reason < 3; ++reason) {
            skip_counts[reason] += group_skip_counts[group * 3 + reason];
        }
    }
}

}  // namespace quant_native
