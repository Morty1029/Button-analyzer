from dataclasses import dataclass
from button_parts.Position import Position


@dataclass
class Indent:
    def __init__(self,
                 margin: Position,
                 padding: Position):
        self.margin = margin
        self.padding = padding

    def __str__(self):
        return f"""
                margin-top: {self.margin.top}px;\n
                margin-right: {self.margin.right}px;\n
                margin-bottom: {self.margin.bottom}px;\n
                margin-left: {self.margin.left}px;\n
                padding-top: {self.margin.top}px;\n
                padding-right: {self.padding.right}px;\n
                padding-bottom: {self.padding.bottom}px;\n
                padding-left: {self.padding.left}px;\n
                """
