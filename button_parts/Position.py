from dataclasses import dataclass


@dataclass
class Position:
    def __init__(self,
                 top: float,
                 right: float,
                 bottom: float,
                 left: float):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

