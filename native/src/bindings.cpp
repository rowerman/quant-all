#include "quant_native/spearman.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void validate_float_array(const py::array& values) {
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

py::tuple pairwise_spearman(
    const py::array& values,
    const py::array& group_offsets,
    int min_n
) {
    validate_float_array(values);
    const auto rows = static_cast<std::size_t>(values.shape(0));
    const auto columns = static_cast<std::size_t>(values.shape(1));
    validate_offsets(group_offsets, rows);
    if (min_n < 5) {
        throw py::value_error("min_n 必须是不小于 5 的整数");
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
            static_cast<std::int64_t*>(count.mutable_data())
        );
    }
    return py::make_tuple(
        std::move(correlation),
        std::move(count)
    );
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
        R"doc(计算按组的完整成对 Spearman 相关矩阵和有效配对数。)doc"
    );
}
