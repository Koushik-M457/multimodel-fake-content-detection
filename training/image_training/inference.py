from PIL import Image
from torchvision import transforms
from .model import get_resnet50


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

