"""以資料類別約束水車讀數；語法為今世 Python，義理取古法。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Power = Literal["river", "ox", "wind"]


@dataclass(frozen=True, slots=True)
class WheelReading:
    turns: int
    buckets: int
    power: Power

    def lifted(self) -> int:
        if self.turns < 0 or self.buckets < 0:
            raise ValueError("轉數與筒數不可為負。")
        return self.turns * self.buckets


def describe_power(power: Power) -> str:
    match power:
        case "river":
            return "堰流激輪"
        case "ox":
            return "牛力運車"
        case "wind":
            return "風帆轉車"


if __name__ == "__main__":
    reading = WheelReading(turns=6, buckets=4, power="river")
    assert reading.lifted() == 24
    assert describe_power(reading.power) == "堰流激輪"
    try:
        WheelReading(-1, 4, "ox").lifted()
    except ValueError:
        pass
    else:
        raise AssertionError("負數須遭拒。")
