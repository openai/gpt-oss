"""未見之題：以雙端佇列求最短路徑，並驗不可達。"""
from collections import deque


def steps(edges: dict[str, list[str]], start: str, goal: str) -> int:
    if start == goal:
        return 0
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        node, distance = queue.popleft()
        for neighbor in edges.get(node, []):
            if neighbor == goal:
                return distance + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return -1


if __name__ == "__main__":
    graph = {"軸": ["輪"], "輪": ["筒", "梘"], "筒": [], "梘": []}
    assert steps(graph, "軸", "梘") == 2
    assert steps(graph, "筒", "軸") == -1
