from PIL import Image
from torchvision import transforms


class ImagePreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def crop_image(input_image_path, output_image_path, coordinates):
        with Image.open(input_image_path) as img:
            cropped_img = img.crop(coordinates)
            cropped_img.save(output_image_path)
            print(f"Cropped image saved to {output_image_path}")
            return cropped_img

    @staticmethod
    def torch_preprocess(path: str):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(path).convert("RGB")
        image = data_transforms(image)
        image = image.unsqueeze(0)
        return image
