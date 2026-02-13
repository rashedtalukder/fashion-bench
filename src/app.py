import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import random

def get_training_data():
    # Download training data from open datasets.
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

def get_test_data():
    # Download test data from open datasets.
    return datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

def load_data(batch_size=64):
    train_dataset = get_training_data()
    test_dataset = get_test_data()

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_loader, test_loader

def create_model(device):

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )

    model.to(device)

    return model

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def get_class_colors():
    """Generate random colors for each class"""
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    colors = {}
    for name in class_names:
        colors[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return class_names, colors

def preprocess_frame(frame):
    """Preprocess camera frame for model input"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor

def run_camera_inference(model, device):
    """Run real-time inference on camera feed"""
    print("Starting camera inference...")
    print("Press 'q' to quit")
    
    class_names, colors = get_class_colors()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    model.eval()
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Preprocess frame
            input_tensor = preprocess_frame(frame).to(device)
            
            # Run inference
            output = model(input_tensor)
            pred_idx = output.argmax(1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][pred_idx].item()
            
            pred_class = class_names[pred_idx]
            color = colors[pred_class]
            
            # Draw bounding box around the frame
            h, w = frame.shape[:2]
            margin = 20
            cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), color, 3)
            
            # Add text with prediction and confidence
            text = f"{pred_class}: {confidence:.2%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (margin, margin - text_height - 10), 
                         (margin + text_width + 10, margin), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (margin + 5, margin - 5), font, 
                       font_scale, (255, 255, 255), thickness)
            
            # Display frame
            cv2.imshow('FashionMNIST Camera Detection', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera inference stopped")

def main():
    model_path = "model.pth"
    accelerator = torch.accelerator.current_accelerator()
    device = accelerator.type

    model = create_model(device)
    
    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Model loaded successfully!")
    else:
        print("No existing model found. Starting training...")
        train_loader, test_loader = load_data()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_loader, model, loss_fn, optimizer, device)
            test(test_loader, model, loss_fn, device)
        print("Done!")
        
        torch.save(model.state_dict(), model_path)
        print(f"Saved PyTorch Model State to {model_path}")
    
    # Run camera inference
    run_camera_inference(model, device)

if __name__ == "__main__":
    main()