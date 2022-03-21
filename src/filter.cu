#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <fmt/core.h>
#include <vector>

#include "common_utils.h"

int main(int argc, char **argv) {
  /**
   * @brief Example of implementing filter
   * Showing usage of cudf's stream Compaction with cudf::ast
   * Simple filter can be done with cudf::transform
   * Complex filter needs help from cudf::ast
   * eg.
   * select * from a, b
   * where a.0 * a.1 > b.0
   */

  auto aCol0 = ghive::MakeColumn<int64_t>({0, 1, 2});
  auto aCol1 = ghive::MakeColumn<int64_t>({4, 3, 3});
  auto bCol0 = ghive::MakeColumn<int64_t>({1, 2, 5});
  auto bCol1 = ghive::MakeColumn<int64_t>({4, 4, 5});

  auto aTableView = cudf::table_view({aCol0->view(), aCol1->view()});
  auto bTableView = cudf::table_view({bCol0->view(), bCol1->view()});
  auto abTableView = cudf::table_view({aTableView, bTableView});

  auto aCol0Ref = cudf::ast::column_reference(0);
  auto aCol1Ref = cudf::ast::column_reference(1);
  auto bCol0Ref = cudf::ast::column_reference(2);

  auto ast0 = cudf::ast::operation(cudf::ast::ast_operator::MUL, aCol0Ref, aCol1Ref);
  auto ast1 = cudf::ast::operation(cudf::ast::ast_operator::GREATER, ast0, bCol0Ref);

  auto maskCol = cudf::compute_column(abTableView, ast1);
  auto filterTable = cudf::apply_boolean_mask(abTableView, maskCol->view());

  // Print Tables
  {
    auto strTable = ghive::ToStrings(filterTable->view());
    ghive::OutputStringifiedTable<3>(strTable);
  }
}
