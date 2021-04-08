import torch
import torch.nn as nn
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
#关于数据集的设置详情见test_9

class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs , dim=1)       #沿dim为1进行拼接   (batch , channel , w , h)


class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.conv1 = torch.nn.Conv2d(1 , 10 , kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88 , 20 , kernel_size=5)   #执行inception后为88维度(24 * 3 + 16)
                                                                #24维三个，16维1个

        self.inception1 = InceptionA(in_channels=10)
        self.inception2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408 , 10)

    def forward(self , x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.inception1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.inception2(x)
        x = x.view(in_size , -1)
        x = self.fc(x)
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
            print('%d , %5d , loss:%.3f'  %(epoch+1 , batch_idx+1 , running_loss/300))
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