"""折半搜尋與拓撲排序：演算法為今術，題材取古器。"""
from collections import defaultdict, deque


def binary_search(sorted_values: list[int], target: int) -> int:
    low, high = 0, len(sorted_values)
    while low < high:
        middle = (low + high) // 2
        if sorted_values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low if low < len(sorted_values) and sorted_values[low] == target else -1


def build_order(parts: dict[str, list[str]]) -> list[str]:
    """parts[構件] 列其先須之構件；回一可行之造器次序。"""
    indegree = {part: len(set(dependencies)) for part, dependencies in parts.items()}
    dependents: dict[str, list[str]] = defaultdict(list)
    for part, dependencies in parts.items():
        for dependency in dependencies:
            if dependency not in indegree:
                raise ValueError("先須之構件未見於籍。")
            dependents[dependency].append(part)
    ready = deque(part for part, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        part = ready.popleft()
        order.append(part)
        for dependent in dependents[part]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if len(order) != len(parts):
        raise ValueError("構件相須成環，不可排次。")
    return order


if __name__ == "__main__":
    values = [1, 3, 3, 5, 8]
    assert binary_search(values, 5) == 3
    assert binary_search(values, 4) == -1
    order = build_order({"輪": ["軸", "筒"], "軸": [], "筒": []})
    assert set(order[:2]) == {"軸", "筒"} and order[-1] == "輪"
    try:
        build_order({"輪": ["軸"], "軸": ["輪"]})
    except ValueError:
        pass
    else:
        raise AssertionError("成環須遭拒。")
