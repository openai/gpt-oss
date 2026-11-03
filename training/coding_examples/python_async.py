"""以非同步之約並舉數算，仍不借外物。"""
import asyncio
from typing import Iterable


async def measure(turns: int, buckets: int) -> int:
    await asyncio.sleep(0)
    if turns < 0 or buckets < 0:
        raise ValueError("轉數與筒數不可為負。")
    return turns * buckets


async def total(readings: Iterable[tuple[int, int]]) -> int:
    tasks = [asyncio.create_task(measure(turns, buckets))
             for turns, buckets in readings]
    values = await asyncio.gather(*tasks)
    return sum(values)


async def guarded_total(readings: Iterable[tuple[int, int]]) -> int:
    async with asyncio.timeout(1):
        return await total(readings)


if __name__ == "__main__":
    assert asyncio.run(guarded_total([(2, 3), (4, 5)])) == 26
    try:
        asyncio.run(guarded_total([(1, 2), (-1, 3)]))
    except ValueError:
        pass
    else:
        raise AssertionError("負數須由工作擲出。")
