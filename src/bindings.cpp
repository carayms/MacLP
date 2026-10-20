#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

// claim core func in aabb_solver.cpp (use raw double pointer, NOT std::vector)
std::vector<std::pair<double, double>> compute_polytope_aabb_cpp(
    int n_dim,
    const std::vector<double>& b_ub,
    const double* A_ptr, // use pointer, avoid raw copy and length_error
    int m_constraints,
    double lb_val,
    double ub_val
);

// Python interface function
py::array_t<double> solve_aabb_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> A_mat,
    py::array_t<double, py::array::c_style | py::array::forcecast> b_vec,
    double lb_val = 0.0,
    double ub_val = 1e30
) {
    // 1. get NumPy buffer storage info
    py::buffer_info A_info = A_mat.request();
    py::buffer_info b_info = b_vec.request();

    // check safe dims
    if (A_info.ndim != 2) {
        throw std::runtime_error("A matrix must be 2-dimensional.");
    }
    if (b_info.ndim != 1) {
        throw std::runtime_error("b vector must be 1-dimensional.");
    }

    int m_constraints = static_cast<int>(A_info.shape[0]);
    int n_dim = static_cast<int>(A_info.shape[1]);

    if (static_cast<int>(b_info.shape[0]) != m_constraints) {
        throw std::runtime_error("Dimension mismatch between A rows and b length.");
    }

    // 2. direct use pointer (0 ram copy)
    const double* A_ptr = static_cast<const double*>(A_info.ptr);
    const double* b_ptr = static_cast<const double*>(b_info.ptr);

    std::vector<double> b_ub(b_ptr, b_ptr + m_constraints);

    // 3. release Python GIL, use multiple CPU 
    py::gil_scoped_release release;

    // 4. Core computation
    auto cpp_results = compute_polytope_aabb_cpp(n_dim, b_ub, A_ptr, m_constraints, lb_val, ub_val);

    // 5. go back GIL, make NumPy return value
    py::gil_scoped_acquire acquire;

    auto result_array = py::array_t<double>({n_dim, 2});
    py::buffer_info res_info = result_array.request();
    double* res_ptr = static_cast<double*>(res_info.ptr);

    for (int i = 0; i < n_dim; ++i) {
        res_ptr[i * 2 + 0] = cpp_results[i].first;  // min
        res_ptr[i * 2 + 1] = cpp_results[i].second; // max
    }

    return result_array;
}

PYBIND11_MODULE(_fast_aabb_cpp, m) {
    m.doc() = "Ultra-fast Polytope AABB Solver using C++, OpenMP, and HiGHS";
    m.def("solve_aabb", &solve_aabb_py, "Compute AABB for Polytope Ax <= b",
          py::arg("A"), 
          py::arg("b"), 
          py::arg("lb_val") = 0.0, 
          py::arg("ub_val") = 1e30);
}