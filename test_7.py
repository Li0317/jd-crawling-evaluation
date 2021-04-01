import numpy as np
import torch
import matplotlib.pyplot as plt

xy = np.loadtxt('diabetes.csv.gz' , delimiter = ','  ,dtype = np.float32)
#delimiter参数为分隔符，此处为","     dtype参数为数据类型，此处为float32(N卡GPU支持32位浮点数)
x_data = torch.from_numpy(xy[: , :-1])      #前面的":"表示所有行，后面的":-1"表示最后一列不要取其他所有列
y_data = torch.from_numpy(xy[: , [-1]])     #前面的":"表示所有行，后面的"[-1]"表示取最后一列
#torch.from_numpy()能创建tensor              #这里的-1必须要加上[]，表示为矩阵，若不加则为向量

class Model (torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8 , 6)       #建立三个线性模型，8维降到6维，6维降到4维，4维降到1维
        self.linear2 = torch.nn.Linear(6 , 4)       #torch.nn.Linear详细见test_5
        self.linear3 = torch.nn.Linear(4 , 1)
        self.sigmoid = torch.nn.Sigmoid()       #sigmoid激活函数，给模型添加一个非线性的变换
        #这里的sigmoid来自nn下，是一个模块，作为一个层，没有参数，只定义一个

    def forward(self , x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()
epoch_list = []
loss_list = []

criterion = torch.nn.BCELoss(reduction='mean')      #reduction参数见rest_5

optimizer = torch.optim.SGD(model.parameters() , lr= 0.01)

for epoch in range(100):
    y_pred = model(x_data)      #这里把所有的数据都放进去了，并没有做mini-batch
    loss = criterion(y_pred , y_data)
    print("epoch:" , epoch , "loss:" , loss.item())

    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()       #清零
    loss.backward()             #反向传播

    optimizer.step()            #更新

print("layer1_w:" , model.linear1.weight)
print("layer1_b" , model.linear1.bias)
print("layer2_w:" , model.linear2.weight)
print("layer2_b" , model.linear2.bias)
print("layer3_w:" , model.linear3.weight)
print("layer3_b" , model.linear3.bias)


plt.plot(epoch_list , loss_list)
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()