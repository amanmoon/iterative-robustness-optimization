import torch
from torchvision import transforms

def predict_car_count(image, model, device='cpu'):
    """
    Takes an image path and a model path, returns the predicted string label.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # (Height, Width)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs