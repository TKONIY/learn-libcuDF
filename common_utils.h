#ifndef LEARN_LIBCUDF_COMMON_UTILS_H
#define LEARN_LIBCUDF_COMMON_UTILS_H

#include <cudf/column/column_view.hpp>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace ghive {

using clk = std::chrono::system_clock;
using ms = std::chrono::milliseconds;
using us = std::chrono::microseconds;
using ns = std::chrono::nanoseconds;
using s = std::chrono::seconds;
using time_point_t = decltype(clk::now());

struct Global {
  constexpr static auto WrongType = "Not-Supported-Type";
};

using StringifiedTable = std::vector<std::vector<std::string>>;
using StringifiedColumn = std::vector<std::string>;

template <typename T> std::unique_ptr<cudf::column> MakeColumn(std::vector<T> &&data) {
  using storeType = T;
  auto rowNum = data.size();
  auto bitmaskAllocBytes = cudf::bitmask_allocation_size_bytes(rowNum);

  rmm::device_buffer resultBitMask{bitmaskAllocBytes, rmm::cuda_stream_default};
  rmm::device_buffer rmmData(data.data(), rowNum * sizeof(storeType),
                             rmm::cuda_stream_default);

  cudf::set_null_mask(static_cast<cudf::bitmask_type *>(resultBitMask.data()), 0,
                      rowNum, true);

  auto col =
      std::make_unique<cudf::column>(cudf::data_type(cudf::type_to_id<T>()), rowNum,
                                     std::move(rmmData), std::move(resultBitMask));
  return col;
};

// only support int, float
template <typename T>
std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, float>, std::vector<T>>
ColumnToVectorNoMask(cudf::column_view const &col) {
  auto vec = std::vector<T>(col.size());
  auto dData = col.data<T>();
  T *hData = vec.data();
  CUDA_TRY(cudaMemcpy(hData, dData, sizeof(T) * col.size(), cudaMemcpyDeviceToHost));
  return vec;
}

// only support int, float
template <typename T>
std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, float>, std::vector<std::string>>
VectorToStringsNoMask(std::vector<T> vec) {
  auto strVec = std::vector<std::string>(vec.size());
  std::transform(vec.begin(), vec.end(), strVec.begin(),
                 [](T val) { return std::to_string(val); });
  return strVec;
}

template <int WIDTH = 5>
void OutputStringifiedTable(const StringifiedTable &table,
                            std::ostream &os = std::cout) {
  assert(!table.empty());
  for (std::size_t row = 0; row < table[0].size(); ++row) {
    for (const auto &col : table) {
      os << std::setw(WIDTH) << col[row] << " ";
    }
    os << std::endl;
  }
};

template <int WIDTH = 5>
void OutputStringifiedColumn(const StringifiedColumn &column,
                             std::ostream &os = std::cout) {
  assert(!column.empty());
  for (const auto &entry : column) {
    os << std::setw(WIDTH) << entry << std::endl;
  }
}

namespace functor {
struct ColumnToStringsNoMaskFunctor {
  template <typename T>
  std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, float>,
                   StringifiedColumn>
  operator()(cudf::column_view const &col) {
    auto vec = ColumnToVectorNoMask<T>(col);
    auto strVec = VectorToStringsNoMask<T>(vec);
    return strVec;
  }

  template <typename T>
  std::enable_if_t<!(std::is_same_v<T, int> || std::is_same_v<T, float>),
                   StringifiedColumn>
  operator()(cudf::column_view const &col) {
    return std::vector<std::string>{static_cast<size_t>(col.size()), Global::WrongType};
  }
};
} // namespace functor
} // namespace ghive
#endif // LEARN_LIBCUDF_COMMON_UTILS_H
