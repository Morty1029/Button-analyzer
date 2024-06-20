import os

from button_detector.ButtonDetector import ButtonDetector
from button_analyzer.ModelTrainer import ModelTrainer
import torch
from button_analyzer.models.ButtonResNet import ButtonResNet
import shutil
from Utils.ImagePreprocessor import ImagePreprocessor


class ButtonAnalyzer:
    def __init__(self):
        self.tmp_dir = "tmp\\"
        self.button_detector = ButtonDetector()
        self.model = ButtonResNet()
        self.model.load_state_dict(torch.load('fitted_models\\resnet101.pth'))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def analyze_buttons_from(self, path) -> list[dict]:
        paths = self.detect_buttons(path)
        images = []
        dict_list = []
        self.model.to(self.device)
        for path in paths:
            images.append(ImagePreprocessor.torch_preprocess(path))
        for image in images:
            dict_list.append(self.model.predict(image.to(self.device)))
        shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)
        return dict_list

    def detect_buttons(self, path) -> list:
        button_paths = []
        bboxes, labels = self.button_detector.get_bboxes_labels(path)
        for label in labels:
            num = label - 1
            path_to_button = self.tmp_dir + str(num) + '.png'
            ImagePreprocessor.crop_image(path,
                                         path_to_button,
                                         (bboxes[num][0], bboxes[num][1], bboxes[num][2], bboxes[num][3]))
            button_paths.append(path_to_button)
        return button_paths
