
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# Image preprocessing function
def preprocess_image(image_path):
    """Preprocess the input image for the ResNet model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Image classification function
def classify_image(image_path):
    """Classify the input image and return the predicted class."""
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(input_tensor)
    _, predicted_class = outputs.max(1)
    return predicted_class.item()

# Example usage
if __name__ == '__main__':
    image_path = "example.jpg"  # Replace with your image path
    predicted_class = classify_image(image_path)
    print(f"Predicted class index: {predicted_class}")
