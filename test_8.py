import torch
from torch.utils.data import Dataset , DataLoader
import numpy as np
import matplotlib.pyplot as plt
'''Dataset是一个抽象类，不能被实例化，只能被其他子类继承     DataLoader可以被实例化'''

class DiabetesDataset(Dataset):
    def __init__(self , filepath):      #获取数据集
        xy = np.loadtxt(filepath , delimiter = ',' , dtype = np.float32)
        self.len = xy.shape[0]      #xy.shape会给出xy这个矩阵的尺寸的元组(行，列)，xy.shape[0]即是行
        self.x_data = torch.from_numpy(xy[: , :-1])
        self.y_data = torch.from_numpy(xy[: , [-1]])

    def __getitem__(self, index):       #获取数据索引
        return self.x_data[index] , self.y_data[index]

    def __len__(self):      #获取数据总量
        return self.len

dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset = dataset , batch_size= 32 , shuffle= True)
'''dataset参数表示传递数据集     batch_size参数表示一个小批量容量是多少    shuffle参数表示是否要打乱数据集、
    num_workers参数表示读数据是否要多线程，为多少则使用多少线程，数据集较小时(小于2W)默认即刻，设置后速度反而降低'''

class Model (torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8 , 6)
        self.linear2 = torch.nn.Linear(6 , 4)
        self.linear3 = torch.nn.Linear(4 , 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self , x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()     #实例化

criterion = torch.nn.BCELoss(reduction='mean')      #loss计算
optimizer = torch.optim.SGD(model.parameters() , lr= 0.01)      #优化器设置
epoch_list = []
loss_list = []

if __name__ == '__main__':      #windows系统下需要将两层循环进行封装，否则会报错
    for epoch in range(100):
        for i , data in enumerate(train_loader , 0):        #enumerate()函数可以为可迭代对象添加一个计数器，记录迭代次数
            inputs , labels = data
            y_pred = model(inputs)

            loss = criterion(y_pred , labels)
            print("epoch:" , epoch , "loss:" , loss.item())
            print("i:" , i)
            epoch_list.append(epoch)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
plt.plot(epoch_list, loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()