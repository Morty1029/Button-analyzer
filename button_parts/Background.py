from dataclasses import dataclass
from possible_values.Colors import Colors


@dataclass
class Background:
    def __init__(self, color: Colors):
        self.color = color

    def __str__(self):
        return f'background-color: {self.color.value};\n'
