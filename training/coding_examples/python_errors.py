"""分內外之失：可料者明示，不可料者不留半成之器。"""
from contextlib import contextmanager
from typing import Iterator


class InvalidWheelState(ValueError):
    """輪之狀態不合所設之界。"""


def validate_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        raise InvalidWheelState("分母不可為零。")
    if numerator < 0 or denominator < 0:
        raise InvalidWheelState("比率之數不可為負。")
    return numerator / denominator


@contextmanager
def checked_calculation() -> Iterator[None]:
    try:
        yield
    except InvalidWheelState as exc:
        raise RuntimeError(f"算器已止：{exc}") from exc
    finally:
        # 釋器之處；此例無外物可釋，仍明其責。
        pass


if __name__ == "__main__":
    assert validate_ratio(6, 3) == 2.0
    with checked_calculation():
        try:
            validate_ratio(1, 0)
        except InvalidWheelState:
            pass
        else:
            raise AssertionError("零分母須遭拒。")
    try:
        with checked_calculation():
            raise InvalidWheelState("不合")
    except RuntimeError:
        pass
    else:
        raise AssertionError("脈絡管理器須改報外失。")
