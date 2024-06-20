from torch import nn
from torchvision import models
from torch.nn.modules.module import T
import torch
import torch.optim as optim
from possible_values.Colors import LabelsColors
from possible_values.FontFamilies import LabelsFontFamilies
from possible_values.FontStyles import LabelsFontStyles
from possible_values.TextAligns import LabelsTextAligns
from possible_values.TextTransforms import LabelsTextTransforms
from possible_values.BorderStyles import LabelsBorderStyles
import copy
from tqdm import tqdm


class ButtonResNet(nn.Module):
    def __init__(self):
        super(ButtonResNet, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fitted = False

        color_classes_num = len(list(LabelsColors))
        f_families_num = len(list(LabelsFontFamilies))
        f_styles_num = len(list(LabelsFontStyles))
        t_aligns_num = len(list(LabelsTextAligns))
        t_transforms_num = len(list(LabelsTextTransforms))
        border_styles_num = len(list(LabelsBorderStyles))

        self.text_color = nn.Linear(num_features, color_classes_num)
        self.f_family = nn.Linear(num_features, f_families_num)
        self.f_style = nn.Linear(num_features, f_styles_num)
        self.text_align = nn.Linear(num_features, t_aligns_num)
        self.text_transform = nn.Linear(num_features, t_transforms_num)
        self.background_color = nn.Linear(num_features, color_classes_num)
        self.border_color = nn.Linear(num_features, color_classes_num)
        self.border_style = nn.Linear(num_features, border_styles_num)

    def forward(self, x):
        x = self.resnet(x)
        return (self.text_color(x),
                self.f_family(x),
                self.f_style(x),
                self.text_align(x),
                self.text_transform(x),
                self.background_color(x),
                self.border_color(x),
                self.border_style(x))

    def fit(self: T, dataloaders: dict, dataset_sizes: dict, num_epochs=30):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        cross_entropy = nn.CrossEntropyLoss()
        best_self_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0

        for epoch in tqdm(range(num_epochs)):
            print()
            print(f'Epoch {epoch}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                correct_color = 0
                correct_f_family = 0
                correct_f_style = 0
                correct_t_align = 0
                correct_t_transform = 0
                correct_background_color = 0
                correct_border_color = 0
                correct_border_style = 0

                for (inputs,
                     l_color,
                     l_f_family,
                     l_f_style,
                     l_t_align,
                     l_t_transform,
                     l_background_color,
                     l_border_color,
                     l_border_style) in dataloaders[phase]:

                    inputs = inputs.to(device)
                    l_color = l_color.to(device)
                    l_f_family = l_f_family.to(device)
                    l_f_style = l_f_style.to(device)
                    l_t_align = l_t_align.to(device)
                    l_t_transform = l_t_transform.to(device)
                    l_background_color = l_background_color.to(device)
                    l_border_color = l_border_color.to(device)
                    l_border_style = l_border_style.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        (l_color_outputs,
                         l_f_family_outputs,
                         l_f_style_outputs,
                         l_t_align_outputs,
                         l_t_transform_outputs,
                         l_background_color_outputs,
                         l_border_color_outputs,
                         l_border_style_outputs) = self(inputs)

                        _, l_color_preds = torch.max(l_color_outputs, 1)
                        _, l_f_family_preds = torch.max(l_f_family_outputs, 1)
                        _, l_f_style_preds = torch.max(l_f_style_outputs, 1)
                        _, l_t_align_preds = torch.max(l_t_align_outputs, 1)
                        _, l_t_transform_preds = torch.max(l_t_transform_outputs, 1)
                        _, l_background_color_preds = torch.max(l_background_color_outputs, 1)
                        _, l_border_color_preds = torch.max(l_border_color_outputs, 1)
                        _, l_border_style_preds = torch.max(l_border_style_outputs, 1)

                        color_loss = cross_entropy(l_color_outputs, l_color)
                        f_family_loss = cross_entropy(l_f_family_outputs, l_f_family)
                        f_style_loss = cross_entropy(l_f_style_outputs, l_f_style)
                        t_align_loss = cross_entropy(l_t_align_outputs, l_t_align)
                        t_transform_loss = cross_entropy(l_t_transform_outputs, l_t_transform)
                        background_color_loss = cross_entropy(l_background_color_outputs, l_background_color)
                        border_color_loss = cross_entropy(l_border_color_outputs, l_border_color)
                        border_style_loss = cross_entropy(l_border_style_outputs, l_border_style)

                        loss = color_loss + f_family_loss + f_style_loss + t_align_loss + t_transform_loss + background_color_loss + border_color_loss + border_style_loss

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    correct_color += torch.sum(l_color_preds == l_color.data)
                    correct_f_family += torch.sum(l_f_family_preds == l_f_family.data)
                    correct_f_style += torch.sum(l_f_style_preds == l_f_style.data)
                    correct_t_align += torch.sum(l_t_align_preds == l_t_align.data)
                    correct_t_transform += torch.sum(l_t_transform_preds == l_t_transform.data)
                    correct_background_color += torch.sum(l_background_color_preds == l_background_color.data)
                    correct_border_color += torch.sum(l_border_color_preds == l_border_color.data)
                    correct_border_style += torch.sum(l_border_style_preds == l_border_style.data)

                num_elements = (dataset_sizes[phase] * dataloaders[phase].batch_size)

                epoch_loss = running_loss / num_elements
                acc_color = correct_color / num_elements
                acc_f_family = correct_f_family / num_elements
                acc_f_style = correct_f_style / num_elements
                acc_t_align = correct_t_align / num_elements
                acc_t_transform = correct_t_transform / num_elements
                acc_background_color = correct_background_color / num_elements
                acc_border_color = correct_border_color / num_elements
                acc_border_style = correct_border_style / num_elements

                print(
                    f'{phase} Loss: {epoch_loss:.4f} '
                    f'Acc color: {acc_color:.4f} '
                    f'Acc font-family: {acc_f_family:.4f} '
                    f'Acc font-style: {acc_f_style:.4f} '
                    f'Acc text-align: {acc_t_align:.4f} '
                    f'Acc text-transform: {acc_t_transform:.4f} '
                    f'Acc background-color: {acc_background_color:.4f} '
                    f'Acc border-color: {acc_border_color:.4f} '
                    f'Acc border-style: {acc_border_style:.4f} ')

                acc_list = [acc_color, acc_border_color, acc_border_style, acc_f_style, acc_t_align, acc_t_transform,
                            acc_f_family, acc_background_color]
                mean_acc = sum(acc_list) / len(acc_list)

                if phase == 'val' and mean_acc > best_acc:
                    best_acc = mean_acc
                    best_self_wts = copy.deepcopy(self.state_dict())

            print()

        print(f'Best val Acc: {best_acc:.4f}')
        self.load_state_dict(best_self_wts)
        self.fitted = True

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def predict(self, image) -> dict:
        with torch.no_grad():
            (l_color_outputs,
             l_f_family_outputs,
             l_f_style_outputs,
             l_t_align_outputs,
             l_t_transform_outputs,
             l_background_color_outputs,
             l_border_color_outputs,
             l_border_style_outputs) = self(image)

        _, l_color_preds = torch.max(l_color_outputs, 1)
        _, l_f_family_preds = torch.max(l_f_family_outputs, 1)
        _, l_f_style_preds = torch.max(l_f_style_outputs, 1)
        _, l_t_align_preds = torch.max(l_t_align_outputs, 1)
        _, l_t_transform_preds = torch.max(l_t_transform_outputs, 1)
        _, l_background_color_preds = torch.max(l_background_color_outputs, 1)
        _, l_border_color_preds = torch.max(l_border_color_outputs, 1)
        _, l_border_style_preds = torch.max(l_border_style_outputs, 1)

        return {
            'color': l_color_preds.item(),
            'font-family': l_f_family_preds.item(),
            'font-style': l_f_style_preds.item(),
            'text-align': l_t_align_preds.item(),
            'text-transform': l_t_transform_preds.item(),
            'background-color': l_background_color_preds.item(),
            'border-color': l_border_color_preds.item(),
            'border-style': l_border_style_preds.item()
        }
