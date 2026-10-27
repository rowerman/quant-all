#include "quant_native/binning.hpp"
#include "quant_native/daily.hpp"
#include "quant_native/spearman.hpp"

#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void validate_matrix(const py::array& values) {
    if (values.ndim() != 2) {
        throw py::value_error("values 必须是二维数组");
    }
    if (!values.dtype().is(py::dtype::of<double>())) {
        throw py::type_error("values 必须是 float64");
    }
    if ((values.flags() & py::array::c_style) == 0) {
        throw py::value_error("values 必须是 C 连续数组");
    }
    if (values.size() == 0) {
        throw py::value_error("values 不能为空");
    }
}

void validate_vector(const py::array& values, const char* name) {
    if (values.ndim() != 1) {
        throw py::value_error(std::string(name) + " 必须是一维数组");
    }
    if (!values.dtype().is(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + " 必须是 float64");
    }
    if ((values.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " 必须是 C 连续数组");
    }
    if (values.size() == 0) {
        throw py::value_error(std::string(name) + " 不能为空");
    }
}

void validate_int64_vector(const py::array& values, const char* name) {
    if (values.ndim() != 1) {
        throw py::value_error(std::string(name) + " 必须是一维数组");
    }
    if (!values.dtype().is(py::dtype::of<std::int64_t>())) {
        throw py::type_error(std::string(name) + " 必须是 int64");
    }
    if ((values.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " 必须是 C 连续数组");
    }
    if (values.size() == 0) {
        throw py::value_error(std::string(name) + " 不能为空");
    }
}

void validate_offsets(
    const py::array& offsets,
    std::size_t rows
) {
    if (offsets.ndim() != 1) {
        throw py::value_error("group_offsets 必须是一维数组");
    }
    if (!offsets.dtype().is(py::dtype::of<std::int64_t>())) {
        throw py::type_error("group_offsets 必须是 int64");
    }
    if ((offsets.flags() & py::array::c_style) == 0) {
        throw py::value_error("group_offsets 必须是 C 连续数组");
    }
    if (offsets.size() < 2) {
        throw py::value_error("group_offsets 至少包含起始和结束边界");
    }
    const auto* data = static_cast<const std::int64_t*>(offsets.data());
    if (data[0] != 0 || data[offsets.size() - 1] != static_cast<std::int64_t>(rows)) {
        throw py::value_error("group_offsets 必须从 0 开始并以样本数结束");
    }
    for (py::ssize_t index = 1; index < offsets.size(); ++index) {
        if (data[index] < data[index - 1]) {
            throw py::value_error("group_offsets 必须单调不减");
        }
    }
}

double validate_n_mad(double n_mad) {
    if (!std::isfinite(n_mad) || n_mad <= 0.0) {
        throw py::value_error("n_mad 必须是有限正数");
    }
    return n_mad;
}

py::tuple pairwise_spearman(
    const py::array& values,
    const py::array& group_offsets,
    int min_n,
    const py::object& target_index
) {
    validate_matrix(values);
    const auto rows = static_cast<std::size_t>(values.shape(0));
    const auto columns = static_cast<std::size_t>(values.shape(1));
    validate_offsets(group_offsets, rows);
    if (min_n < 5) {
        throw py::value_error("min_n 必须是不小于 5 的整数");
    }
    std::optional<std::size_t> target;
    if (!target_index.is_none()) {
        if (!PyLong_Check(target_index.ptr()) || PyBool_Check(target_index.ptr())) {
            throw py::type_error("target_index 必须是非负整数或 None");
        }
        const auto parsed = py::cast<std::int64_t>(target_index);
        if (parsed < 0 || static_cast<std::size_t>(parsed) >= columns) {
            throw py::value_error("target_index 超出矩阵列范围");
        }
        target = static_cast<std::size_t>(parsed);
    }

    const auto group_count =
        static_cast<std::size_t>(group_offsets.size() - 1);
    py::array_t<double> correlation(
        {
            static_cast<py::ssize_t>(group_count),
            static_cast<py::ssize_t>(columns),
            static_cast<py::ssize_t>(columns),
        }
    );
    py::array_t<std::int64_t> count(
        {
            static_cast<py::ssize_t>(group_count),
            static_cast<py::ssize_t>(columns),
            static_cast<py::ssize_t>(columns),
        }
    );
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::pairwise_spearman(
            static_cast<const double*>(values.data()),
            rows,
            columns,
            static_cast<const std::int64_t*>(group_offsets.data()),
            group_count,
            min_n,
            threads,
            static_cast<double*>(correlation.mutable_data()),
            static_cast<std::int64_t*>(count.mutable_data()),
            target
        );
    }
    return py::make_tuple(
        std::move(correlation),
        std::move(count)
    );
}

py::array_t<double> daily_rank_pct(
    const py::array& values,
    const py::array& group_offsets
) {
    validate_vector(values, "values");
    const auto rows = static_cast<std::size_t>(values.shape(0));
    validate_offsets(group_offsets, rows);
    const auto group_count =
        static_cast<std::size_t>(group_offsets.size() - 1);
    py::array_t<double> result(static_cast<py::ssize_t>(rows));
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::daily_rank_pct(
            static_cast<const double*>(values.data()),
            rows,
            static_cast<const std::int64_t*>(group_offsets.data()),
            group_count,
            threads,
            static_cast<double*>(result.mutable_data())
        );
    }
    return result;
}

py::tuple daily_pearson(
    const py::array& left,
    const py::array& right,
    const py::array& group_offsets,
    int min_n
) {
    validate_vector(left, "left");
    validate_vector(right, "right");
    if (left.shape(0) != right.shape(0)) {
        throw py::value_error("left 与 right 长度必须一致");
    }
    const auto rows = static_cast<std::size_t>(left.shape(0));
    validate_offsets(group_offsets, rows);
    if (min_n < 2) {
        throw py::value_error("min_n 必须是不小于 2 的整数");
    }
    const auto group_count =
        static_cast<std::size_t>(group_offsets.size() - 1);
    py::array_t<double> correlation(
        static_cast<py::ssize_t>(group_count)
    );
    py::array_t<std::int64_t> count(
        static_cast<py::ssize_t>(group_count)
    );
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::daily_pearson(
            static_cast<const double*>(left.data()),
            static_cast<const double*>(right.data()),
            rows,
            static_cast<const std::int64_t*>(group_offsets.data()),
            group_count,
            min_n,
            threads,
            static_cast<double*>(correlation.mutable_data()),
            static_cast<std::int64_t*>(count.mutable_data())
        );
    }
    return py::make_tuple(
        std::move(correlation),
        std::move(count)
    );
}

py::array_t<double> daily_winsorize_mad(
    const py::array& values,
    const py::array& group_offsets,
    double n_mad
) {
    validate_vector(values, "values");
    const double multiplier = validate_n_mad(n_mad);
    const auto rows = static_cast<std::size_t>(values.shape(0));
    validate_offsets(group_offsets, rows);
    const auto group_count =
        static_cast<std::size_t>(group_offsets.size() - 1);
    py::array_t<double> result(static_cast<py::ssize_t>(rows));
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::daily_winsorize_mad(
            static_cast<const double*>(values.data()),
            rows,
            static_cast<const std::int64_t*>(group_offsets.data()),
            group_count,
            multiplier,
            threads,
            static_cast<double*>(result.mutable_data())
        );
    }
    return result;
}

py::array_t<double> daily_zscore(
    const py::array& values,
    const py::array& group_offsets
) {
    validate_vector(values, "values");
    const auto rows = static_cast<std::size_t>(values.shape(0));
    validate_offsets(group_offsets, rows);
    const auto group_count =
        static_cast<std::size_t>(group_offsets.size() - 1);
    py::array_t<double> result(static_cast<py::ssize_t>(rows));
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::daily_zscore(
            static_cast<const double*>(values.data()),
            rows,
            static_cast<const std::int64_t*>(group_offsets.data()),
            group_count,
            threads,
            static_cast<double*>(result.mutable_data())
        );
    }
    return result;
}

py::array_t<double> daily_winsorize_zscore(
    const py::array& values,
    const py::array& group_offsets,
    double n_mad
) {
    validate_vector(values, "values");
    const double multiplier = validate_n_mad(n_mad);
    const auto rows = static_cast<std::size_t>(values.shape(0));
    validate_offsets(group_offsets, rows);
    const auto group_count =
        static_cast<std::size_t>(group_offsets.size() - 1);
    py::array_t<double> result(static_cast<py::ssize_t>(rows));
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::daily_winsorize_zscore(
            static_cast<const double*>(values.data()),
            rows,
            static_cast<const std::int64_t*>(group_offsets.data()),
            group_count,
            multiplier,
            threads,
            static_cast<double*>(result.mutable_data())
        );
    }
    return result;
}

py::array_t<std::int64_t> daily_quantile_bins(
    const py::array& values,
    const py::array& group_offsets,
    int quantiles
) {
    validate_vector(values, "values");
    const auto rows = static_cast<std::size_t>(values.shape(0));
    validate_offsets(group_offsets, rows);
    if (quantiles < 2) {
        throw py::value_error("quantiles 必须是不小于 2 的整数");
    }
    const auto group_count =
        static_cast<std::size_t>(group_offsets.size() - 1);
    py::array_t<std::int64_t> result(static_cast<py::ssize_t>(rows));
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::daily_quantile_bins(
            static_cast<const double*>(values.data()),
            rows,
            static_cast<const std::int64_t*>(group_offsets.data()),
            group_count,
            quantiles,
            threads,
            static_cast<std::int64_t*>(result.mutable_data())
        );
    }
    return result;
}

py::tuple bin_returns_multi(
    const py::array& values,
    const py::array& day_codes,
    const py::array& bin_codes,
    const py::array& weights,
    int day_count,
    int bin_count
) {
    validate_matrix(values);
    validate_int64_vector(day_codes, "day_codes");
    validate_int64_vector(bin_codes, "bin_codes");
    validate_vector(weights, "weights");
    const auto rows = static_cast<std::size_t>(values.shape(0));
    const auto labels = static_cast<std::size_t>(values.shape(1));
    if (
        day_codes.size() != static_cast<py::ssize_t>(rows)
        || bin_codes.size() != static_cast<py::ssize_t>(rows)
        || weights.size() != static_cast<py::ssize_t>(rows)
    ) {
        throw py::value_error("values、day_codes、bin_codes 与 weights 行数必须一致");
    }
    if (day_count < 1 || bin_count < 1) {
        throw py::value_error("day_count 与 bin_count 必须为正整数");
    }
    const auto days = static_cast<std::size_t>(day_count);
    const auto bins = static_cast<std::size_t>(bin_count);
    py::array_t<double> returns(
        {
            static_cast<py::ssize_t>(days),
            static_cast<py::ssize_t>(bins),
            static_cast<py::ssize_t>(labels),
        }
    );
    py::array_t<double> pools(
        {
            static_cast<py::ssize_t>(days),
            static_cast<py::ssize_t>(labels),
        }
    );
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::bin_returns_multi(
            static_cast<const double*>(values.data()),
            rows,
            labels,
            static_cast<const std::int64_t*>(day_codes.data()),
            static_cast<const std::int64_t*>(bin_codes.data()),
            static_cast<const double*>(weights.data()),
            days,
            bins,
            threads,
            static_cast<double*>(returns.mutable_data()),
            static_cast<double*>(pools.mutable_data())
        );
    }
    return py::make_tuple(std::move(returns), std::move(pools));
}

py::tuple bin_turnover(
    const py::array& values,
    int quantiles
) {
    validate_matrix(values);
    if (quantiles < 1) {
        throw py::value_error("quantiles 必须为正整数");
    }
    const auto rows = static_cast<std::size_t>(values.shape(0));
    const auto columns = static_cast<std::size_t>(values.shape(1));
    py::array_t<double> per_bin(
        {
            static_cast<py::ssize_t>(rows),
            static_cast<py::ssize_t>(quantiles),
        }
    );
    py::array_t<double> overall(static_cast<py::ssize_t>(rows));
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::bin_turnover(
            static_cast<const double*>(values.data()),
            rows,
            columns,
            quantiles,
            threads,
            static_cast<double*>(per_bin.mutable_data()),
            static_cast<double*>(overall.mutable_data())
        );
    }
    return py::make_tuple(std::move(per_bin), std::move(overall));
}

py::tuple daily_row_spearman(
    const py::array& left,
    const py::array& right,
    int min_n
) {
    validate_matrix(left);
    validate_matrix(right);
    if (
        left.shape(0) != right.shape(0)
        || left.shape(1) != right.shape(1)
    ) {
        throw py::value_error("left 与 right 形状必须一致");
    }
    if (min_n < 2) {
        throw py::value_error("min_n 必须是不小于 2 的整数");
    }
    const auto rows = static_cast<std::size_t>(left.shape(0));
    const auto columns = static_cast<std::size_t>(left.shape(1));
    py::array_t<double> correlation(static_cast<py::ssize_t>(rows));
    py::array_t<std::int64_t> count(static_cast<py::ssize_t>(rows));
    const int threads = quant_native::threads_from_environment();

    {
        py::gil_scoped_release release;
        quant_native::daily_row_spearman(
            static_cast<const double*>(left.data()),
            static_cast<const double*>(right.data()),
            rows,
            columns,
            min_n,
            threads,
            static_cast<double*>(correlation.mutable_data()),
            static_cast<std::int64_t*>(count.mutable_data())
        );
    }
    return py::make_tuple(std::move(correlation), std::move(count));
}

}  // namespace

PYBIND11_MODULE(quant_native, module) {
    module.doc() = "量化项目批量 C++ 计算核";
    module.def(
        "pairwise_spearman",
        &pairwise_spearman,
        py::arg("values"),
        py::arg("group_offsets"),
        py::arg("min_n") = 5,
        py::arg("target_index") = py::none(),
        R"doc(计算按组的成对 Spearman 相关矩阵；指定 target_index 时只计算目标行。)doc"
    );
    module.def(
        "daily_rank_pct",
        &daily_rank_pct,
        py::arg("values"),
        py::arg("group_offsets"),
        R"doc(按组计算并列平均名次分位。)doc"
    );
    module.def(
        "daily_pearson",
        &daily_pearson,
        py::arg("left"),
        py::arg("right"),
        py::arg("group_offsets"),
        py::arg("min_n") = 5,
        R"doc(按组计算有限配对 Pearson 相关系数和有效配对数。)doc"
    );
    module.def(
        "daily_winsorize_mad",
        &daily_winsorize_mad,
        py::arg("values"),
        py::arg("group_offsets"),
        py::arg("n_mad") = 3.0,
        R"doc(按组使用中位数和 MAD 裁剪极值。)doc"
    );
    module.def(
        "daily_zscore",
        &daily_zscore,
        py::arg("values"),
        py::arg("group_offsets"),
        R"doc(按组计算样本标准差口径的截面 z-score。)doc"
    );
    module.def(
        "daily_winsorize_zscore",
        &daily_winsorize_zscore,
        py::arg("values"),
        py::arg("group_offsets"),
        py::arg("n_mad") = 3.0,
        R"doc(按组融合 MAD 去极值和截面 z-score。)doc"
    );
    module.def(
        "daily_quantile_bins",
        &daily_quantile_bins,
        py::arg("values"),
        py::arg("group_offsets"),
        py::arg("quantiles"),
        R"doc(按组计算平均并列名次分位组，返回从 0 开始的组号。)doc"
    );
    module.def(
        "bin_returns_multi",
        &bin_returns_multi,
        py::arg("values"),
        py::arg("day_codes"),
        py::arg("bin_codes"),
        py::arg("weights"),
        py::arg("day_count"),
        py::arg("bin_count"),
        R"doc(一次计算多标签分位收益和池内均值。)doc"
    );
    module.def(
        "bin_turnover",
        &bin_turnover,
        py::arg("values"),
        py::arg("quantiles"),
        R"doc(按完整交易日矩阵计算全池与各分位换手。)doc"
    );
    module.def(
        "daily_row_spearman",
        &daily_row_spearman,
        py::arg("left"),
        py::arg("right"),
        py::arg("min_n") = 5,
        R"doc(按行对共同有限样本重新排名并计算 Spearman。)doc"
    );
}
