"""以集合、計數與分組摹布爾揀類之法。"""
from collections import Counter, defaultdict
from collections.abc import Iterable


def common_kind(left: set[str], right: set[str]) -> set[str]:
    return left & right


def idempotent(kinds: set[str]) -> bool:
    return common_kind(kinds, kinds) == kinds


def tally(symbols: Iterable[str]) -> Counter[str]:
    return Counter(symbols)


def group_by_power(readings: Iterable[tuple[str, int]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for power, turns in readings:
        grouped[power].append(turns)
    return dict(grouped)


if __name__ == "__main__":
    water = {"筒", "梘", "輪"}
    assert idempotent(water)
    assert common_kind(water, {"輪", "帆"}) == {"輪"}
    assert tally("水水風牛")["水"] == 2
    assert group_by_power([("river", 2), ("ox", 3), ("river", 4)]) == {
        "river": [2, 4], "ox": [3],
    }
