#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <fmt/core.h>
#include <vector>

#include "common_utils.h"

int main(int argc, char **argv) {
  /**
   * @brief Example of ghive select operator
   * Showing usage of cudf::ast to do math opeartion
   * select col1*col2+2*col0
   * from table;
   */

  auto col0 = ghive::MakeColumn<int64_t>({4, 5, 6});
  auto col1 = ghive::MakeColumn<int64_t>({0, 1, 2});
  auto col2 = ghive::MakeColumn<int64_t>({1, 2, 3});

  auto table = cudf::table_view({col0->view(), col1->view(), col2->view()});
  auto ref0 = cudf::ast::column_reference(0);
  auto ref1 = cudf::ast::column_reference(1);
  auto ref2 = cudf::ast::column_reference(2);

  auto literal_value = cudf::numeric_scalar<int64_t>(2);
  auto literal_2 = cudf::ast::literal(literal_value);
  auto ast0 = cudf::ast::operation(cudf::ast::ast_operator::MUL, ref1, ref2);
  auto ast1 = cudf::ast::operation(cudf::ast::ast_operator::MUL, ref0, literal_2);
  auto ast2 = cudf::ast::operation(cudf::ast::ast_operator::ADD, ast0, ast1);

  auto before = ghive::clk::now(); // timer start
  auto result = cudf::compute_column(table, ast2);
  auto after = ghive::clk::now(); // timer stop

  auto duration = after - before;
  auto us = std::chrono::duration_cast<ghive::us>(duration).count();

  fmt::print(
      "select col1*col2+2*col0 from table with: {} columns {} rows, cost {} us\n",
      table.num_columns(), table.num_rows(), us);

  // Print Tables
  {
    auto strTable = ghive::ToStrings(cudf::table_view({result->view()}));
    ghive::OutputStringifiedTable<3>(strTable);
  }
}
