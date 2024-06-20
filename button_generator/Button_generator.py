from button_parts.Background import Background, Colors
from button_parts.Border import Border, BorderStyles
from button_parts.Button import Button
from button_parts.Font import Font, FontFamilies, FontStyles, TextAligns, TextTransforms
from button_parts.Indent import Indent
from button_parts.Position import Position
from button_parts.Size import Size
import random


class ButtonGenerator:

    @staticmethod
    def buttons_generate(num) -> list[Button]:
        buttons = []
        for _ in range(num):
            buttons.append(ButtonGenerator.button_generate())
        return buttons

    @staticmethod
    def button_generate() -> Button:
        size = ButtonGenerator.size_generate()
        font = ButtonGenerator.font_generate(size)
        back = ButtonGenerator.background_generate()
        while font.color == back.color:
            back.color = random.choice(list(Colors))
        border = ButtonGenerator.border_generate()
        indent = ButtonGenerator.indent_generate()
        button = Button(size=size,
                        font=font,
                        background=back,
                        border=border,
                        indent=indent)
        return button

    @staticmethod
    def get_button_from_file(path) -> Button:
        # TODO
        pass

    @staticmethod
    def background_generate() -> Background:
        color = random.choice(list(Colors))
        return Background(color=color)

    @staticmethod
    def border_generate() -> Border:
        color = random.choice(list(Colors))
        style = random.choice(list(BorderStyles))
        return Border(color=color, style=style)

    @staticmethod
    def font_generate(size: Size) -> Font:
        color = random.choice(list(Colors))
        f_family = random.choice(list(FontFamilies))
        f_style = random.choice(list(FontStyles))
        f_weight = random.randint(1, 1000)
        t_align = random.choice(list(TextAligns))
        t_transforms = random.choice(list(TextTransforms))
        f_size = random.randint(20, 100)
        max_font_size = int(min(size.height // 2, size.width // 10))  # Adjust the factors as needed
        f_size = random.randint(10, max_font_size)
        return Font(color=color,
                    font_family=f_family,
                    font_style=f_style,
                    font_weight=f_weight,
                    text_align=t_align,
                    text_transform=t_transforms,
                    font_size=f_size)

    @staticmethod
    def indent_generate() -> Indent:
        margin = ButtonGenerator.position_generate()
        padding = ButtonGenerator.position_generate()
        return Indent(margin=margin, padding=padding)

    @staticmethod
    def position_generate() -> Position:
        top = random.randint(0, 100)
        left = random.randint(0, 100)
        bottom = random.randint(0, 100)
        right = random.randint(0, 100)
        return Position(top=top,
                        left=left,
                        bottom=bottom,
                        right=right)

    @staticmethod
    def size_generate() -> Size:
        height = random.randint(100, 400)
        width = random.randint(100, 400)
        return Size(height=height,
                    width=width)
