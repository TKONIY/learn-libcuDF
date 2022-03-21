#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
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
  auto keyCol = ghive::MakeColumn<int64_t>({4, 3, 3});
  auto valueCol = ghive::MakeColumn<int64_t>({1, 2, 5});

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = valueCol->view();
  requests[0].aggregations.push_back(std::move(agg));
  cudf::groupby::groupby gby(cudf::table_view({keyCol->view()}));

  auto before = ghive::clk::now(); // timer start
  auto results = gby.aggregate(requests);
  auto after = ghive::clk::now(); // timer stop

  auto duration = after - before;
  auto us = std::chrono::duration_cast<ghive::us>(duration).count();

  fmt::print("group by cost {} us\n", us);

  // Print Tables
  auto keyColAfter = results.first->view().column(0);
  auto valColAfter = results.second.at(0).results.at(0)->view();
  fmt::print("value column type: {}", valColAfter.type().id());
  auto tableAfter = cudf::table_view({keyColAfter, valColAfter});

  ghive::OutputStringifiedTable(ghive::ToStrings(tableAfter));
}
