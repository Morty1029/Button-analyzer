from torchvision import transforms
from button_analyzer.ButtonsDataset import ButtonsDataset
from button_analyzer.models.ButtonResNet import ButtonResNet


class ModelTrainer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.button_dataset = ButtonsDataset(transform=self.transform)
        self.resnet = ButtonResNet()

    def start_train(self):
        train, test = self.button_dataset.get_loaders()
        dataloaders = {
            'train': train,
            'val': test
        }
        dataset_sizes = {
            'train': len(train),
            'val': len(test)
        }
        self.resnet.fit(dataloaders, dataset_sizes)
        self.resnet.save_model("fitted_models\\resnet101.pth")
