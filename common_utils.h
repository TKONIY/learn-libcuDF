#ifndef LEARN_LIBCUDF_COMMON_UTILS_H
#define LEARN_LIBCUDF_COMMON_UTILS_H

#include <string>
#include <vector>
#include <cudf/column/column_view.hpp>
#include <ostream>
#include <iomanip>

namespace ghive {
    struct Global {
        constexpr static auto WrongType = "Not-Supported-Type";
    };

    using StringifiedTable = std::vector<std::vector<std::string>>;


    // only support int, float
    template<typename T>
    std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, float>, std::vector<T>>
    ColumnToVectorNoMask(cudf::column_view const &col) {
        auto vec = std::vector<T>(col.size());
        auto dData = col.data<T>();
        T *hData = vec.data();
        CUDA_TRY(cudaMemcpy(hData, dData, sizeof(T) * col.size(), cudaMemcpyDeviceToHost));
        return vec;
    }

    // only support int, float
    template<typename T>
    std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, float>, std::vector<std::string>>
    VectorToStringsNoMask(std::vector<T> vec) {
        auto strVec = std::vector<std::string>(vec.size());
        std::transform(vec.begin(), vec.end(), strVec.begin(), [](T val) {
            return std::to_string(val);
        });
        return strVec;
    }


    template<int WIDTH = 5>
    void OutputStringifiedTable(const StringifiedTable &table, std::ostream &os) {
        assert(!table.empty());
        for (std::size_t row = 0; row < table[0].size(); ++row) {
            for (const auto &col: table) {
                os << std::setw(WIDTH) << col[row] << " ";
            }
            os << std::endl;
        }
    };

    namespace functor {
        struct ColumnToStringsNoMaskFunctor {
            template<typename T>
            std::enable_if_t<std::is_same_v<T, int> || std::is_same_v<T, float>, std::vector<std::string>>
            operator()(cudf::column_view const &col) {
                auto vec = ColumnToVectorNoMask<T>(col);
                auto strVec = VectorToStringsNoMask<T>(vec);
                return strVec;
            }

            template<typename T>
            std::enable_if_t<!(std::is_same_v<T, int> || std::is_same_v<T, float>), std::vector<std::string>>
            operator()(cudf::column_view const &col) {
                return std::vector<std::string>{static_cast<size_t>(col.size()), Global::WrongType};
            }

        };
    }
}
#endif //LEARN_LIBCUDF_COMMON_UTILS_H
