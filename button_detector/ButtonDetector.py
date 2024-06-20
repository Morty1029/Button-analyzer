from ultralytics import YOLO
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union


class ButtonDetector:
    def __init__(self):
        self.model: YOLO = YOLO('button_detector\\models\\best.pt')

    def detect_button(self, path_to_img: str = 'res\\pic\\0.png'):
        # TODO
        bboxes, labels = self.get_bboxes_labels(path_to_img)
        print(labels)
        print(bboxes)
        pass

    def get_bboxes_labels(self, path_to_img: str = 'res\\pic\\0.png'):
        result = self.model.predict(path_to_img, device='cuda:0')[0].boxes
        bboxes = result.xyxy.cpu().numpy()
        labels = result.cls.cpu().numpy().astype(int).tolist()
        return bboxes, labels

    def get_height_width(self, path_to_img: str = 'res\\pic\\0.png'):
        bboxes, labels = self.get_bboxes_labels(path_to_img)
        height = bboxes[0][3] - bboxes[0][1]
        width = bboxes[0][2] - bboxes[0][0]
        return height, width


