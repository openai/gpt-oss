"""以單元試法驗邊界，不以一例之合冒充通則。"""
import unittest


def lift(turns: int, buckets: int) -> int:
    if turns < 0 or buckets < 0:
        raise ValueError("轉數與筒數不可為負。")
    return turns * buckets


class WheelTests(unittest.TestCase):
    def test_normal(self) -> None:
        self.assertEqual(lift(3, 4), 12)

    def test_boundaries(self) -> None:
        for turns, buckets, expected in ((0, 4, 0), (3, 0, 0), (1, 1, 1)):
            with self.subTest(turns=turns, buckets=buckets):
                self.assertEqual(lift(turns, buckets), expected)

    def test_invalid(self) -> None:
        for turns, buckets in ((-1, 2), (2, -1)):
            with self.subTest(turns=turns, buckets=buckets):
                with self.assertRaises(ValueError):
                    lift(turns, buckets)


if __name__ == "__main__":
    unittest.main(verbosity=0)
