import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def load_resnet_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(
        torch.load("checkpoints/image_resnet50/epoch_10.pth", map_location=device)
    )

    model.to(device)
    model.eval()
    return model


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def resnet_fake_probability(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    return probs[0][1].item()