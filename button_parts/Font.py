from dataclasses import dataclass
from possible_values.Colors import Colors
from possible_values.FontFamilies import FontFamilies
from possible_values.FontStyles import FontStyles
from possible_values.TextAligns import TextAligns
from possible_values.TextTransforms import TextTransforms


@dataclass
class Font:
    def __init__(self,
                 color: Colors,
                 font_family: FontFamilies,
                 font_style: FontStyles,
                 font_weight: int,
                 text_align: TextAligns,
                 text_transform: TextTransforms,
                 font_size: int):
        self.color = color
        self.font_family = font_family
        self.font_style = font_style
        self.font_weight = font_weight - font_weight % 100
        self.text_align = text_align
        self.text_transform = text_transform
        self.font_size = font_size

    def __str__(self):
        return f"""
            color: {self.color.value};\n
            font-family: {self.font_family.value};\n
            font-style: {self.font_style.value};\n
            font-weight: {round(self.font_weight, 2)};\n
            text-align: {self.text_align.value};\n
            text_transform: {self.text_transform.value};\n
            font-size: {self.font_size};\n
            """
