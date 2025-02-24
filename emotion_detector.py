import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import cv2
import numpy as np
from PIL import Image
import time

# Define emotion labels
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define transformation for images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FER2013 dataset
data_dir = "./Test and Train"  # Change this to dataset location
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(EMOTION_CLASSES))  # Modify output layer

# Move model to GPU (if available), or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to device (GPU or CPU)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "emotion_model.pth")
    print("Model saved!")

# Train the model
train_model(model, train_loader, val_loader, epochs=10)

# Real-Time Emotion Detection
def detect_emotions():
    print("Loading model...")
    model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
    model.eval()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is successfully opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    else:
        print("Webcam opened successfully.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Move to the same device
            
            with torch.no_grad():
                output = model(face_tensor)
                pred = torch.argmax(output).item()
                label = EMOTION_CLASSES[pred]

            # Draw rectangle around face and add emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the image with detected faces and emotions
        cv2.imshow('Emotion Detector', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
        
        # Add a slight delay to avoid maxing out the CPU
        time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()

# Run real-time emotion detection
detect_emotions()
