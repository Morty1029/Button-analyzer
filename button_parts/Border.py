from dataclasses import dataclass
from possible_values.Colors import Colors
from possible_values.BorderStyles import BorderStyles
from possible_values.Radii import Radii


@dataclass
class Border:
    def __init__(self, color: Colors, style: BorderStyles):
        self.color = color
        self.style = style

    def __str__(self):
        return f"""
               border-color: {self.color.value};\n
               border-style: {self.style.value};\n
               """
