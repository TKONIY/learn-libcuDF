#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
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
  auto leftCol1 = ghive::MakeColumn<int64_t>({0, 2, 1, 2, 4, 5});
  auto rightCol1 = ghive::MakeColumn<int64_t>({4, 3, 3, 3, 3, 3});
  auto rightCol2 = ghive::MakeColumn<int64_t>({1, 2, 1, 1, 1, 5});

  auto buildTableView = cudf::table_view({leftCol1->view()});
  auto buildKeyView = cudf::table_view({leftCol1->view()});
  auto probeTableView = cudf::table_view({rightCol1->view(), rightCol2->view()});
  auto probeKeyView = cudf::table_view({rightCol2->view()});

  auto beforeJoin = ghive::clk::now(); // timer start

  cudf::hash_join hashJoin(buildKeyView, cudf::null_equality::EQUAL);
  auto outputSize = hashJoin.inner_join_size(probeKeyView);
  auto [probeTableIdxs, buildTableIdxs] =
      hashJoin.inner_join(probeKeyView, cudf::null_equality::EQUAL, outputSize);

  // gather
  auto probeTableGatherMap = cudf::column_view{
      cudf::data_type{cudf::type_id::INT32}, // cudf::size_type = int32_t
      static_cast<cudf::size_type>(probeTableIdxs->size()), probeTableIdxs->data()};

  auto buildTableGatherMap = cudf::column_view{
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(buildTableIdxs->size()), buildTableIdxs->data()};

  auto buildTableAfterJoin = cudf::gather(buildTableView, buildTableGatherMap);
  auto probeTableAfterJoin = cudf::gather(probeTableView, probeTableGatherMap);

  auto afterJoin = ghive::clk::now(); // timer stop

  auto durationJoin = afterJoin - beforeJoin;
  auto usJoin = std::chrono::duration_cast<ghive::us>(durationJoin).count();

  // fmt::print("joinTable with: {} columns, cost {} us\n", result->num_columns(),
  // usJoin);

  // Print Tables
  {
    auto strBuildTable = ghive::ToStrings(buildTableAfterJoin->view());
    auto strProbeTable = ghive::ToStrings(probeTableAfterJoin->view());
    fmt::print("Build Table:\n");
    ghive::OutputStringifiedTable<3>(strBuildTable);
    fmt::print("Probe Table:\n");
    ghive::OutputStringifiedTable<3>(strProbeTable);
  }
}
