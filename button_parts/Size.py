from dataclasses import dataclass


@dataclass
class Size:

    def __init__(self,
                 height: float,
                 width: float):
        self.height = height
        self.width = width

    def __str__(self):
        return f"""
                height: {self.height}px;\n
                width: {self.width}px;\n
                """
