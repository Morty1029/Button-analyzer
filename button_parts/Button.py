import os
from button_parts.Size import Size
from button_parts.Font import Font
from button_parts.Border import Border
from button_parts.Background import Background
from button_parts.Indent import Indent
from html2image import Html2Image


class Button:
    def __init__(self,
                 size: Size,
                 font: Font,
                 background: Background,
                 border: Border,
                 indent: Indent):
        """Предусматривается измерение всех размеров в пикселях"""
        self.size = size
        self.font = font
        self.background = background
        self.border = border
        self.indent = indent

    def __str__(self):
        format_string = f"""
                        {self.size.__str__()}
                        {self.font.__str__()}
                        {self.background.__str__()}
                        {self.border.__str__()}
                        {self.indent.__str__()}
                        """
        return '{\n' + format_string + '}'

    def save_to_file(self, path=None):
        if path is None:
            labels_dir = 'labels'
            if not os.path.exists(labels_dir):
                os.makedirs(labels_dir)
            num_tmp_files = len(
                [name for name in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, name))])
            filename = f'button_{num_tmp_files}.txt'
            path = os.path.join(labels_dir, filename)
        with open(path, 'w+') as file:
            file.write(self.__str__())
            file.close()

    def visualize(self, path=None):
        html = """
                <!DOCTYPE html>
                    <html>
                     <head>
                      <meta charset="utf-8">
                      <title>Buttons</title>
                      <style>
                        .btn
                      </style>
                     </head>
                     <body>
                      <p><button class="btn">Button</button></p>
                     </body>
                    </html>
                    """
        html_code = html.replace('.btn', '.btn' + self.__str__())
        pic_dir = 'pic'
        if path is None:
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            num_tmp_files = len([name for name in os.listdir(pic_dir) if os.path.isfile(os.path.join(pic_dir, name))])
            path = f'button_{num_tmp_files}.png'

        hti = Html2Image()
        hti.output_path = pic_dir
        hti.screenshot(html_str=html_code, save_as=path)
