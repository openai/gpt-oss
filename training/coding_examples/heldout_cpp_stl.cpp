// 未見之題：以標準容器與演算法驗上下界。
#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <vector>

int main() {
    std::vector<int> turns{2, 4, 6, 8};
    const auto it = std::lower_bound(turns.begin(), turns.end(), 5);
    assert(it != turns.end() && *it == 6);

    std::map<std::string, int> ledger{{"river", 3}, {"ox", 2}};
    ledger["wind"] = 1;
    assert(ledger.at("wind") == 1);
    assert(std::all_of(turns.begin(), turns.end(), [](int value) { return value > 0; }));
    return 0;
}
