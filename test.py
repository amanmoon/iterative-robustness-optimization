import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import ClassifierCNN_128p

DATA_PATH = './misclassified'
MODEL_PATH = 'model/car_counter_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=DATA_PATH, transform=data_transforms)

test_loader = DataLoader(test_dataset, shuffle=False)

model = ClassifierCNN_128p().to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)

total = 0
correct = 0
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test images: {100 * correct / total:.2f}%")