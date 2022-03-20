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
  auto leftCol1 = ghive::MakeColumn<int>({0, 1, 2});
  auto rightCol1 = ghive::MakeColumn<int>({4, 3, 3});
  auto rightCol2 = ghive::MakeColumn<int>({1, 2, 5});

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
    auto strVectors = ghive::StringifiedTable();
    for (cudf::size_type i = 0; i < joinTable->num_columns(); ++i) {
      auto columnView = joinTable->get_column(i).view();

      fmt::print("col{} type = {} row number = {}\n", i,
                 static_cast<int32_t>(columnView.type().id()), columnView.size());

      auto vec = cudf::type_dispatcher(columnView.type(),
                                       ghive::functor::ColumnToStringsNoMaskFunctor{},
                                       columnView);
      strVectors.push_back(vec);
    }

    ghive::OutputStringifiedTable<3>(strVectors);
  }

  /**
   * @brief Example of cudf::groupby
   * keys:      rightCol1
   * values:    rightCol2
   * 
   * table
   * 4 1
   * 3 2
   * 3 5
   *
   * result
   * 4 1
   * 3 7
   *
   * select sum(col1)
   * from table
   * group by col0
   */

  auto agg = cudf::make_sum_aggregation<cudf::groupby_aggregation>();

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests[0].values = rightCol2->view();
  requests[0].aggregations.push_back(std::move(agg));

  cudf::groupby::groupby gby(cudf::table_view({rightCol1->view()}));

  auto results = gby.aggregate(requests);
}
