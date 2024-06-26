from util import Vec, argv
import math
import sys


def test_vec():
    vec = Vec(0.13, 4)
    assert vec.x == 0.13 and vec.y == 4

    copy = Vec(vec)
    assert vec is not copy
    assert vec == copy

    copy += Vec(0.87, 0)
    assert vec != copy
    assert copy.x == 1

    assert Vec(1, 2) + Vec(3, 2) == Vec(4, 4)
    assert Vec(1, 2) - Vec(3, 2) == Vec(-2, 0)
    assert Vec(1, 2) * 2 == Vec(2, 4)
    assert Vec(1, 2) / 2 == Vec(0.5, 1)

    assert Vec(3, -2).cross(Vec(1, 2)) == 8
    assert Vec(3, -2).dot(Vec(1, 2)) == -1

    assert Vec.from_angle(0).round(3) == Vec(1, 0)
    assert Vec.from_angle(math.pi / 2).round(3) == Vec(0, 1)

    assert abs(Vec(3, 4)) == 5
    assert Vec(1, 0).tolist() == (1, 0)
    assert Vec(0, -3).normalized() == Vec(0, -1)
    assert Vec(1, 0).rotated(math.pi / 2).round(3) == Vec(0, 1)


def test_argv():
    sys.argv.extend(
        [
            "--int",
            "51",
            "--float",
            "12.42",
            "--str",
            "string",
            "--bool1",
            "True",
            "--bool2",
            "False",
        ]
    )

    assert argv("none", "") == ""
    assert argv("int", 0) == 51
    assert argv("float", 0.0) == 12.42
    assert argv("str", "") == "string"
    assert argv("bool1", False) is True
    assert argv("bool2", True) is False
