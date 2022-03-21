#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <fmt/core.h>
#include <vector>

#include "common_utils.h"

int main(int argc, char **argv) {
  /**
   * @brief Example of cudf::inner_join
   * Showing usage of cudf::table and cudf::column.
   * Also give a good practice of cudf::type_dispatcher.
   */
  auto leftCol1 = ghive::MakeColumn<int64_t>({0, 1, 2});
  auto rightCol1 = ghive::MakeColumn<int64_t>({4, 3, 3});
  auto rightCol2 = ghive::MakeColumn<int64_t>({1, 2, 5});

  auto leftTableView = cudf::table_view({leftCol1->view()});
  auto rightTableView = cudf::table_view({rightCol1->view(), rightCol2->view()});

  auto beforeJoin = ghive::clk::now(); // timer start
  auto joinTable = cudf::inner_join(leftTableView, rightTableView, {0}, {1});
  auto afterJoin = ghive::clk::now(); // timer stop

  auto durationJoin = afterJoin - beforeJoin;
  auto usJoin = std::chrono::duration_cast<ghive::us>(durationJoin).count();

  fmt::print("joinTable with: {} columns, cost {} us\n", joinTable->num_columns(),
             usJoin);

  // Print Tables
  {
    auto strTable = ghive::ToStrings(joinTable->view());
    ghive::OutputStringifiedTable<3>(strTable);
  }
}
