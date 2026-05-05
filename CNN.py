import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import Accuracy, Precision, Recall, F1Score
import time
import matplotlib.pyplot as plt
import random as r

class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.Flatten = nn.Flatten()
        self.ReLU = nn.ReLU()
        self.CNN_1 = nn.Conv2d(1, 8, 4)
        
        # Фиктивный вход для определения размера
        dummy = torch.randn(1, 1, 28, 28)   # (batch=1, channels=1, 28x28)
        out = self.CNN_1(dummy)
        num_features = out.view(1, -1).size(1)
        
        self.Linear_2 = nn.Linear(num_features, num_features//4)
        self.Linear_3 = nn.Linear(num_features//4, 10)

    def forward(self, x):
        x = self.ReLU(self.CNN_1(x))
        x = self.Flatten(x)
        x = self.ReLU(self.Linear_2(x))
        x = self.Linear_3(x)
        
        return x
    
def load_data(root="./data"):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    test_dataset  = datasets.MNIST(root=root, train=False, download=False, transform=transform)
    
    return train_dataset, test_dataset

def train_model(model: nn.Module, train_data: DataLoader, num_epochs: int = 30):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    model.train()
    
    start = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for images, labels in train_data:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_data.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} – Train loss: {avg_loss:.4f}")
        
        if (epoch == num_epochs//2) or epoch == num_epochs - 1:
            torch.save(model, f"./Models/CNN(3)_b{train_data.batch_size}_e{epoch+1}.pth")
        
    end = time.time()
    
    print(f"Время обучения (с): {round(end - start)}")

def test_model(model: nn.Module, test_data: DataLoader, show_result: bool = False):
    metrics = {
    "Accuracy":  Accuracy(task="multiclass", num_classes=10),
    "Precision": Precision(task="multiclass", average='macro', num_classes=10),
    "Recall":    Recall(task="multiclass", average='macro', num_classes=10),
    "F1":        F1Score(task="multiclass", average='macro', num_classes=10)
    }
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_data:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            for m in metrics.values():
                m.update(probs, labels)
            
            if show_result:    
                rand_idx = r.randint(0, len(images)-1)
                model_y = torch.argmax(probs[rand_idx], dim=0)
                plt.figure()                     
                plt.imshow(images[rand_idx][0], cmap="gray")
                plt.title(f"Result model: {model_y}")
                plt.show(block=False)

    print("\nTest metrics:")
    for name, metric in metrics.items():
        print(f"{name}: {metric.compute():.4f}")

def normal_predict(model: nn.Module, x: torch.Tensor):
        model.eval()
        
        outputs = model(x)
        
        probs = torch.softmax(outputs, dim=1)
        
        predicted_classes = torch.argmax(probs, dim=1)
        
        return predicted_classes.item()

def validate_own_data(model: nn.Module, show_result: bool = False):    
    with torch.no_grad():
        X = torch.load("./Tensors images/Batch1_X.pt")
        y = torch.load("./Tensors images/Batch1_Y.pt")
        summary = 0
        idx = 0
    
        for image in X:
            label = normal_predict(model=model, x=image.unsqueeze(0).unsqueeze(0))
            
            if label == y[idx]:
                summary += 1
            
            idx += 1
            
            if show_result:
                plt.figure()                     
                plt.imshow(image, cmap="gray")
                plt.title(f"Result model: {label}")
                plt.show(block=False)
        
        print(f"Accuracy: {round(summary/len(y),2)}")
   
if __name__ == "__main__":
    # train_ds, test_ds = load_data()
    # train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    # test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False)
    
    # model = CNN_MNIST()

    # train_model(model=model, train_data=train_loader, num_epochs=50)
    # test_model(model=model, test_data=test_loader)
    
    model = torch.load("./Models/CNN(3)_b64_e50.pth", weights_only=False)
    validate_own_data(model=model, show_result=True)
    # test_model(model=model, test_data=test_loader)