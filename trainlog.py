import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

# Directory paths for training, validation, and testing
train_dir = 'Real Life Violence Dataset'
val_dir = 'Real Life Violence Dataset'
test_dir = 'Real Life Violence Dataset'

# Define the image transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to fit Inception's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Use ImageFolder to load the datasets
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

# Create DataLoader for batching and shuffling
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load the pre-trained Inception model
model = models.inception_v3(pretrained=True)

# Freeze layers (optional for transfer learning)
for param in model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer for binary classification (Violence vs. Non-Violence)
model.fc = nn.Linear(model.fc.in_features, 1)  # Output: 1 for binary classification (violence or non-violence)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        predicted = torch.round(torch.sigmoid(outputs))  # Convert logits to probabilities
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Validation step
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())

            val_loss += loss.item()

            # Calculate accuracy
            predicted = torch.round(torch.sigmoid(outputs))  # Convert logits to probabilities
            val_correct_predictions += (predicted == labels).sum().item()
            val_total_predictions += labels.size(0)

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = val_correct_predictions / val_total_predictions

        print(f"Validation - Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}")

# Saving the trained model
torch.save(model.state_dict(), 'violence_detection_model.pth')

# Evaluate on the test set
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.round(torch.sigmoid(outputs))  # Convert logits to probabilities

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Generate classification report
print(classification_report(all_labels, all_predictions))

# Confusion matrix
print(confusion_matrix(all_labels, all_predictions))
