from torch.utils.data import Dataset, random_split
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image


class ButtonsDataset(Dataset):
    def __init__(self,
                 img_dir: str = 'res\\cut_pictures_v2',
                 annotations_file: str = 'res\\csv_data\\labels_data_v3.csv',
                 transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + '.png')
        image = Image.open(img_path).convert("RGB")
        color = self.img_labels.iloc[idx, 3]
        font_family = self.img_labels.iloc[idx, 4]
        font_style = self.img_labels.iloc[idx, 5]
        text_align = self.img_labels.iloc[idx, 7]
        text_transform = self.img_labels.iloc[idx, 8]
        background_color = self.img_labels.iloc[idx, 9]
        border_color = self.img_labels.iloc[idx, 10]
        border_style = self.img_labels.iloc[idx, 11]
        if self.transform:
            image = self.transform(image)
        return (image,
                torch.tensor(color),
                torch.tensor(font_family),
                torch.tensor(font_style),
                torch.tensor(text_align),
                torch.tensor(text_transform),
                torch.tensor(background_color),
                torch.tensor(border_color),
                torch.tensor(border_style),
                )

    def get_loaders(self, train_size=0.8, batch_size=64):
        tr_size = int(train_size * len(self))
        test_size = len(self) - tr_size
        train_dataset, test_dataset = random_split(self, [tr_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=15)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=15)
        return train_loader, test_loader


