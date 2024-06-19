from __future__ import annotations

import math

Number = float | int


class Vec:
    def __init__(self, x: "Vec" | Number = 0, y: Number = 0):
        if isinstance(x, Number):
            self.x = x
            self.y = y
        else:
            self.x, self.y = x

    def __repr__(self):
        return f"Vec({self.x}, {self.y})"

    def __getitem__(self, index):
        return (self.x, self.y)[index]

    def __setitem__(self, index, value):
        if index == 0:
            self.x = value
            return
        if index == 1:
            self.y = value
            return
        raise IndexError("Index out of range, expected 0 to 1")

    def __iter__(self):
        yield self.x
        yield self.y

    def __eq__(self, other: "Vec"):
        return self.x == other.x and self.y == other.y

    def __add__(self, other: "Vec"):
        return Vec(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec"):
        return Vec(self.x - other.x, self.y - other.y)

    def __mul__(self, value: Number):
        return Vec(self.x * value, self.y * value)

    def __rmul__(self, value: Number):
        return Vec(value * self.x, value * self.y)

    def __truediv__(self, value: Number):
        return Vec(self.x / value, self.y / value)

    def __rtruediv__(self, value: Number):
        return Vec(value / self.x, value / self.y)

    def __floordiv__(self, value: Number):
        return Vec(self.x // value, self.y // value)

    def __neg__(self):
        return Vec(-self.x, -self.y)

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @staticmethod
    def from_angle(angle: float):
        return Vec(math.cos(angle), math.sin(angle))

    def rotated(self, angle: float):
        length = abs(self)
        angle = math.atan2(self.y, self.x) + angle
        x = math.cos(angle) * length
        y = math.sin(angle) * length
        return Vec(x, y)

    def normalized(self):
        length = abs(self)
        return Vec(self.x / length, self.y / length)

    def round(self, digits: int=0):
        return Vec(round(self.x, digits), round(self.y, digits))

    def dot(self, other: "Vec"):
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec"):
        """
        Returns the z value of the cross product of the two vectors in 3d space.
        A cross product in 2d space is undefined.
        """
        return (self.x * other.y) - (self.y * other.x)

    def tolist(self):
        return self.x, self.y
