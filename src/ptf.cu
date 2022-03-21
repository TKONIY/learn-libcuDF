#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <fmt/core.h>
#include <vector>

#include "common_utils.h"

int main(int argc, char **argv) {
  /**
   * @brief Example of Hive PTF operator
   * RANK() over (PARTITION BY column)
   * cudf::groupby::groupby::scan()
   */

  auto keyCol = ghive::MakeColumn<int64_t>({4, 3, 3, 5, 5, 5, 3});
  auto valueCol = ghive::MakeColumn<int64_t>({1, 2, 3, 4, 5, 2, 5});

  // Example 1:
  // implement RANK:
  // Firstly sort key and values. Then use make_rank_aggregation to generate rank
  // RANK() needs presort
  {
    auto sortValue = cudf::table_view({keyCol->view(), valueCol->view()});
    auto sortKey = cudf::table_view({keyCol->view(), valueCol->view()}); // rank
    auto sortedTable = cudf::sort_by_key(sortValue, sortKey);
    ghive::OutputStringifiedTable(ghive::ToStrings(sortedTable->view()));
    auto sortedKeyCol = sortedTable->view().column(0);
    auto sortedValueCol = sortedTable->view().column(1);

    auto agg = cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>();
    std::vector<cudf::groupby::scan_request> requests;
    requests.emplace_back(cudf::groupby::scan_request());
    requests[0].values = sortedValueCol;
    requests[0].aggregations.push_back(std::move(agg));
    cudf::groupby::groupby gby(cudf::table_view({sortedKeyCol}),
                               cudf::null_policy::EXCLUDE, cudf::sorted::YES);

    auto before = ghive::clk::now(); // timer start
    auto results = gby.scan(requests);
    auto after = ghive::clk::now(); // timer stop

    auto duration = after - before;
    auto us = std::chrono::duration_cast<ghive::us>(duration).count();

    fmt::print("ptf cost {} us\n", us);

    // Print Tables
    auto keyColAfter = results.first->view().column(0);
    auto valColAfter = results.second.at(0).results.at(0)->view();
    fmt::print("value column type: {}", valColAfter.type().id());
    auto tableAfter = cudf::table_view({keyColAfter, valColAfter});

    ghive::OutputStringifiedTable(ghive::ToStrings(tableAfter));
  }

  // Example 2:
  // implement SUM() over (PARTITION BY)
  // without order by
  // haven't find other solutions. 1. aggregate; 2. join
  {
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

    fmt::print("ptf cost {} us\n", us);

    // Print Tables
    auto keyColAfter = results.first->view().column(0);
    auto valColAfter = results.second.at(0).results.at(0)->view();
    fmt::print("value column type: {}", valColAfter.type().id());
    auto tableAfterGroupBy = cudf::table_view({keyColAfter, valColAfter});

    // join; Can replace with inner_join
    auto originTable = cudf::table_view({keyCol->view(), valueCol->view()});
    auto joinTable = cudf::inner_join(originTable, tableAfterGroupBy, {0}, {0});
    auto withSumTable = joinTable->select({0, 1, 3});
    //
    ghive::OutputStringifiedTable(ghive::ToStrings(withSumTable));
  }
}
