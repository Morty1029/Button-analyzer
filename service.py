import pandas as pd
import numpy as np
from button_detector.ButtonDetector import ButtonDetector
from possible_values.Colors import LabelsColors, Colors
from possible_values.BorderStyles import LabelsBorderStyles, BorderStyles
from possible_values.FontStyles import LabelsFontStyles, FontStyles
from possible_values.FontFamilies import LabelsFontFamilies, FontFamilies
from possible_values.TextAligns import LabelsTextAligns, TextAligns
from possible_values.TextTransforms import LabelsTextTransforms, TextTransforms
from PIL import Image
import os
from tqdm import tqdm
import torch


def rewrite_data(path):
    data = pd.read_csv(path)
    data = data[[
        'button_number',
        'height',
        'width',
        'color',
        'font-family',
        'font-style',
        'font-weight',
        'text-align',
        'text-transform',
        'background-color',
        'border-color',
        'border-style'
    ]]
    data['text-transform'] = data['text-transform'].replace(0.0, 'none')
    data['text-transform'] = data['text-transform'].apply(lambda x: LabelsTextTransforms[TextTransforms(x).name].value)
    data['color'] = data['color'].apply(lambda x: LabelsColors[Colors(x).name].value)
    data['font-family'] = data['font-family'].apply(lambda x: LabelsFontFamilies[FontFamilies(x).name].value)
    data['font-style'] = data['font-style'].apply(lambda x: LabelsFontStyles[FontStyles(x).name].value)
    data['text-align'] = data['text-align'].apply(lambda x: LabelsTextAligns[TextAligns(x).name].value)
    data['background-color'] = data['background-color'].apply(lambda x: LabelsColors[Colors(x).name].value)
    data['border-color'] = data['border-color'].apply(lambda x: LabelsColors[Colors(x).name].value)
    data['border-style'] = data['border-style'].apply(lambda x: LabelsBorderStyles[BorderStyles(x).name].value)

    data.to_csv("res\\csv_data\\labels_data_v3.csv", index=False)


def rename_buttons():
    path = 'res\\pic'
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        os.rename(file_path, os.path.join(path, file + '.png'))


def report(path):
    data = pd.read_csv(path)
    print(data.describe())


def crop_image(input_image_path, output_image_path, coordinates):
    with Image.open(input_image_path) as img:
        cropped_img = img.crop(coordinates)
        cropped_img.save(output_image_path)
        print(f"Cropped image saved to {output_image_path}")


def extract_button(input_path):
    img = Image.open(input_path).convert("RGBA")
    img_data = np.array(img)

    non_transparent_pixels = np.where(img_data[:, :, 3] > 0)
    top, left = np.min(non_transparent_pixels, axis=1)
    bottom, right = np.max(non_transparent_pixels, axis=1)

    button_image = img.crop((left, top, right + 1, bottom + 1))

    return button_image


def cuda():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    rewrite_data("res\\csv_data\\labels_data_v2.csv")
    '''input_path = 'res\\pic\\'
    output_path = 'res\\cut_pictures_v2\\'
    pictures = os.listdir(input_path)
    for picture in tqdm(pictures):
        button_image = extract_button(input_path + picture)
        button_image.save(output_path + picture)'''
    # print(cuda())

