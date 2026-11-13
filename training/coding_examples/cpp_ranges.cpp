// 以範圍視圖揀數、變形而後合計；只取標準函式庫。
#include <algorithm>
#include <cassert>
#include <numeric>
#include <ranges>
#include <vector>

int main() {
    namespace views = std::ranges::views;
    auto values = views::iota(1, 9)
        | views::transform([](int value) { return value * value; })
        | views::filter([](int value) { return value % 2 == 0; });

    std::vector<int> collected;
    for (int value : values) {
        collected.push_back(value);
    }
    assert((collected == std::vector<int>{4, 16, 36, 64}));
    assert(std::accumulate(collected.begin(), collected.end(), 0) == 120);

    auto first = std::ranges::find(collected, 16);
    assert(first != collected.end());
    return 0;
}
