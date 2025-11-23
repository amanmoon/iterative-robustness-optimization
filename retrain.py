import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

from model import ClassifierCNN_128p

DATA_PATH = './dataset_train'
DATA_AUG_PATH = './misclassified'
MODEL_PATH = 'model/car_counter_model.pth'
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on: {DEVICE}")

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=DATA_PATH, transform=data_transforms)
dataset_aug = datasets.ImageFolder(root=DATA_AUG_PATH, transform=data_transforms)

full_dataset = ConcatDataset([train_dataset, dataset_aug])

total_size = len(full_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training Size: {total_size}")

model = ClassifierCNN_128p().to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model/car_counter_model_1.pth")
print("Model saved as car_counter_model.pth")