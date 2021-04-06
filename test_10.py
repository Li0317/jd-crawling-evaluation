import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize((0.1307 , ) , (0.3081 , ))])

train_dataset = datasets.MNIST(root='../datasets/mnist' , train=True , download=True , transform = transform)
train_loader = DataLoader(train_dataset , shuffle=True , batch_size=batch_size)

test_dataset = datasets.MNIST(root='../datasets/mnist' , train=False , download=True , transform=transform)
test_loader = DataLoader(test_dataset , shuffle=False , batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1 , 10 , kernel_size=5)    #卷积核为5*5，输入通道为1，输出通道为10
        self.conv2 = torch.nn.Conv2d(10 , 20 , kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        '''MAX pooling指的是对于每一个channel（假设有N个channel），
           将该channel的feature map的像素值选取其中最大值作为该channel的代表，
           从而得到一个N维向量表示。
           Maxpool没有参数需要更新，只需要设置一个'''
        self.fc1 = torch.nn.Linear(320 , 160)
        self.fc2 = torch.nn.Linear(160 , 80)
        self.fc3 = torch.nn.Linear(80 , 10)

    def forward(self , x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))     #先做卷积，再做pooling(池化)，再做激活
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size , -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters() , lr=0.01 , momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx , data in enumerate(train_loader , 0):
        inputs , target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs , target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx %300 == 299:
            print('%d , %5d , loss:%.3f'  %(epoch+1 , batch_idx+1 , running_loss/2000))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images , labels = data
            outputs = model(images)
            _ , predicted = torch.max(outputs.data , dim=1)

            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print("Accuracy on test set:%d %%" % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()