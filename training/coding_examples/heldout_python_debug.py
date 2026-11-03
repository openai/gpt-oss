"""未見之題：改正邊界與累計之誤，並自驗。"""
from collections.abc import Iterable


def cumulative_lift(readings: Iterable[tuple[int, int]]) -> list[int]:
    total = 0
    result = []
    for turns, buckets in readings:
        if turns < 0 or buckets < 0:
            raise ValueError("讀數不可為負。")
        total += turns * buckets
        result.append(total)
    return result


def first_at_least(values: list[int], threshold: int) -> int:
    for index, value in enumerate(values):
        if value >= threshold:
            return index
    return -1


if __name__ == "__main__":
    assert cumulative_lift([(1, 2), (3, 4), (0, 9)]) == [2, 14, 14]
    assert first_at_least([2, 14, 14], 14) == 1
    assert first_at_least([2, 14], 15) == -1
