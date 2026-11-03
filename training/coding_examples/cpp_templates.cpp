// 以樣版與概念約束整數算器；今世 C++ 語法，古器為喻。
#include <cassert>
#include <concepts>
#include <type_traits>
#include <vector>

template <std::integral T>
constexpr T bounded_lift(T turns, T buckets) {
    if (turns < 0 || buckets < 0) {
        return 0;
    }
    return turns * buckets;
}

template <typename T>
concept Readable = requires(T value) {
    { value.turns } -> std::convertible_to<int>;
};

template <Readable T>
int sum_lifts(const std::vector<T>& readings) {
    int total = 0;
    for (const auto& reading : readings) {
        total += bounded_lift(reading.turns, reading.buckets);
    }
    return total;
}

struct Reading { int turns; int buckets; };

static_assert(bounded_lift(3, 4) == 12);
static_assert(std::is_same_v<decltype(bounded_lift(2, 2)), int>);

int main() {
    const std::vector<Reading> readings{{2, 3}, {4, 1}};
    assert(sum_lifts(readings) == 10);
    assert(bounded_lift(-1, 4) == 0);
    return 0;
}
