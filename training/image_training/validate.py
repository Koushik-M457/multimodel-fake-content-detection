import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from preprocessing.image_preprocess import val_test_transform
from model import get_resnet50

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_resnet50().to(device)
model.load_state_dict(torch.load("checkpoints/image/resnet50_best.pth"))
model.eval()

test_data = ImageFolder("data/image/processed/test", transform=val_test_transform)
test_loader = DataLoader(test_data, batch_size=32)

correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Test Accuracy:", correct / total)
