#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

#include "common_utils.h"

template<typename T>
std::unique_ptr<cudf::column> MakeColumn(std::vector<T> &&data) {
    using storeType = T;
    auto rowNum = data.size();
    auto bitmaskAllocBytes = cudf::bitmask_allocation_size_bytes(rowNum);

    rmm::device_buffer resultBitMask{bitmaskAllocBytes, rmm::cuda_stream_default};
    rmm::device_buffer rmmData(data.data(), rowNum * sizeof(storeType), rmm::cuda_stream_default);

    cudf::set_null_mask(static_cast<cudf::bitmask_type *>(resultBitMask.data()), 0, rowNum, true);

    auto col = std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_to_id<T>()),
            rowNum,
            std::move(rmmData),
            std::move(resultBitMask)
    );
    return col;
};

int main(int argc, char **argv) {
    /**
     * @brief Example of cudf::inner_join
     * Showing usage of cudf::table and cudf::column.
     * Also give a good practice of cudf::type_dispatcher.
     */
    auto leftCol1 = MakeColumn<int>({0, 1, 2});
    auto rightCol1 = MakeColumn<int>({4, 9, 3});
    auto rightCol2 = MakeColumn<int>({1, 2, 5});

    auto leftTableView = cudf::table_view({leftCol1->view()});
    auto rightTableView = cudf::table_view({rightCol1->view(), rightCol2->view()});

    auto joinTable = cudf::inner_join(leftTableView, rightTableView, {0}, {1});

    std::cout << "joinTable with: " << joinTable->num_columns() << " columns: " << std::endl;

    auto strVectors = ghive::StringifiedTable();
    for (cudf::size_type i = 0; i < joinTable->num_columns(); ++i) {
        auto columnView = joinTable->get_column(i).view();
        std::cout << "col" << i
                  << " type = " << static_cast<int32_t>(columnView.type().id())
                  << " column number = " << columnView.size()
                  << std::endl;
        auto vec = cudf::type_dispatcher(columnView.type(),
                                         ghive::functor::ColumnToStringsNoMaskFunctor{},
                                         columnView);
        strVectors.push_back(vec);
    }

    ghive::OutputStringifiedTable<3>(strVectors, std::cout);
    return 0;
}
