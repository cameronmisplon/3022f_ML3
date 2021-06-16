import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image

training_data = datasets.MNIST(
        root="./",
        train=True,
        download=False,
        transform=ToTensor()
)
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
device = "cuda" if torch.cuda.is_available() else "cpu"

class ANNnetwork(nn.Module):
    def __init__(self): #creates the structure for the neural network with 28*28,512,512,10 neurons in each layer respectively. Each neuron makes use of the ReLU activation function
        super(ANNnetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.ReLU()
        )
    def forward(self, x): #specifies how data will pass through the neural network
        x = self.flatten(x)
        logits =self.linear_relu_stack(x)
        return logits

def train(dataloader, model, lossFunction, optimization): # adjusts the weights of the neural network using crossentropy loss function as well as performing the backpropogation
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = lossFunction(pred, y)
        optimization.zero_grad()
        loss.backward()
        optimization.step()
        if batch % 100 ==0:
            loss, current = loss.item(), batch*len(X)

model = ANNnetwork().to(device)
lossFunction = nn.CrossEntropyLoss()
optimization = torch.optim.SGD(model.parameters(), lr=0.001)
epochs = 10
for i in range(epochs):
    print(f"Epoch {i+1}\n----------------------")
    train(train_dataloader,model,lossFunction,optimization)

print("Done!")
img_path = input("Please enter a filepath:\n")
while img_path != "exit": #Handles making predictions on user inputs
    image = Image.open(img_path)
    image = ToTensor()(image).unsqueeze(0)
    image.to(device)
    pred = model(image)
    print("Classifier: "+str(pred.argmax(1).type(torch.float).sum().item()))
    img_path = input("Please enter a filepath:\n")
