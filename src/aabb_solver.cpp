#include <iostream>
#include <vector>
#include <utility>
#include <omp.h>
#include "Highs.h"

std::vector<std::pair<double, double>> compute_polytope_aabb_cpp(
    int n_dim,
    const std::vector<double>& b_ub,
    const double* A_flat, // use pointer
    int m_constraints,
    double lb_val,
    double ub_val
) {
    std::vector<std::pair<double, double>> results(n_dim);
    std::vector<int> col_indices(n_dim);
    for (int i = 0; i < n_dim; ++i) col_indices[i] = i;

    #pragma omp parallel
    {
        Highs highs;
        highs.setOptionValue("output_flag", false);
        //highs.setOptionValue("presolve", "on");
        highs.setOptionValue("presolve", "off");
        highs.setOptionValue("solver", "simplex"); // using simplex

        HighsLp lp;
        lp.num_col_ = n_dim;
        lp.num_row_ = 0;
        lp.col_lower_ = std::vector<double>(n_dim, lb_val);
        lp.col_upper_ = std::vector<double>(n_dim, ub_val);
        lp.col_cost_  = std::vector<double>(n_dim, 0.0);

        highs.passModel(lp);

        for (int r = 0; r < m_constraints; ++r) {
            const double* row_data = A_flat + (r * n_dim);
            highs.addRow(-kHighsInf, b_ub[r], n_dim, col_indices.data(), row_data);
        }

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_dim; ++i) {
            
            // --- A. Min x_i ---
            highs.changeColCost(i, 1.0);
            highs.run();

            double l_i = -kHighsInf;
            HighsModelStatus status_min = highs.getModelStatus();
            if (status_min == HighsModelStatus::kOptimal) {
                l_i = highs.getSolution().col_value[i];
            } else {
                l_i = lb_val;
            }

            // --- B. Max x_i ---
            highs.changeColCost(i, -1.0);
            highs.run();

            double u_i = kHighsInf;
            HighsModelStatus status_max = highs.getModelStatus();
            if (status_max == HighsModelStatus::kOptimal) {
                u_i = highs.getSolution().col_value[i];
            } else {
                u_i = ub_val;
            }

            highs.changeColCost(i, 0.0);
            results[i] = {l_i, u_i};
        }
    } 

    return results;
}