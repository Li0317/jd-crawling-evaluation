import torch
from torchvision import transforms  #处理图象的工具
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F     #激活函数使用relu()
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize((0.1307 , ) , (0.3081 , ))])
''' transforms.Compose()能将[]里多个transforms组合起来使用                          ↑mean(均值)   ↑std(标准差)
    transforms.ToTensor()能将图象变为一个Tensor(c * w * h)
    c为channel(通道) ， w为宽 ， h为高 ； 同时把图象的像素值由0到255变为0到1
    transforms.Normalize()给定均值：(R,G,B) 标准差：（R，G，B）用均值和标准偏差对张量图像进行归一化(切换到0，1分布)，
    即：Normalized_image=(image-mean)/std，这种变换不支持PIL图像
    对于n通道，这种变换将使输入pytorch的每个通道规范化。
'''
#MNIST数据集是28 * 28的图象
train_dataset = datasets.MNIST(root='../datasets/mnist' , train=True
                               , download=True , transform=transform)
train_loader = DataLoader(train_dataset , shuffle=True , batch_size=batch_size)

test_dataset = datasets.MNIST(root='../datasets/mnist' , train=False
                               , download=True , transform=transform)
test_loader = DataLoader(test_dataset , shuffle=False , batch_size=batch_size)  #训练集不打乱数据集

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(784 , 512)
        self.layer2 = torch.nn.Linear(512 , 256)
        self.layer3 = torch.nn.Linear(256 , 128)
        self.layer4 = torch.nn.Linear(128 , 64)
        self.layer5 = torch.nn.Linear(64 , 10)

    def forward(self , x):
        x = x.view(-1 , 784)    #view()能改变张量形状，这里转换为二阶张量(矩阵)，这里的-1能自动算出有多少张量
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)   #最后一层不做激活

model = Net()

criterion = torch.nn.CrossEntropyLoss()     #交叉熵损失
optimizer = optim.SGD(model.parameters() , lr= 0.01 , momentum=0.5)     #冲量值为0.5

def train(epoch):
    running_loss = 0.0
    for batch_idx , data in enumerate(train_loader , 0):
        inputs , target = data      #x输入到inputs里 ， y输入到target里
        optimizer.zero_grad()       #优化器在优化之前清零

        outputs = model(inputs)
        loss = criterion(outputs , target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()     #累计损失，取loss值时用item()，防止构建计算图
        if batch_idx %300 == 299:   #每训练300次输出一次
            print("[%d , %5d] loss: %.3f" % (epoch+1 , batch_idx+1 , running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0     #正确数
    total = 0       #总数
    with torch.no_grad():       #不需要计算梯度
        for data in test_loader:
            images , labels = data
            outputs = model(images)
            _ , predicted = torch.max(outputs.data , dim=1) #dim参数表示维度，这里为1即代表沿着维度1(行)找max
                                                            #torch.max返回两个值(最大值，最大值下标)
            total += labels.size(0) #labels为一个N*1的矩阵,labels.size为一个元组，取第0个元素即为N
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set: %d %%" % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        if epoch %5 == 4:
            test()
