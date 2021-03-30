import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#可能会报错OMP: Error #15:
# 若报错则加入上面两行
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0] , [2.0] , [3.0]])
y_data = torch.Tensor([[0] , [0] , [1]])

class LogisticRegressionModel(torch.nn.Module):         #逻辑斯蒂回归模块
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self , x):
        y_pred = F.sigmoid(self.linear(x))      #F.sigmoid为σ(x) = 1/(1+ｅ^(-x))，其值在[0,1]，为饱和函数
        return y_pred

model = LogisticRegressionModel()

criterion = torch.nn.BCELoss(size_average=False)       #BCE损失计算，loss = -(ylog y_pred + (1-y)log(1-y_pred))
                                # size_average为True时求均值，False不求均值

optimizer = torch.optim.SGD(model.parameters() , lr= 0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred , y_data)

    print("epoch:" , epoch , "loss:" , loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w = " , model.linear.weight.item())
print("b = " , model.linear.bias.item())

x = np.linspace(0 , 10 , 200)   #从0到10等距取200个
x_test = torch.Tensor(x).view((200,1))      #生成200行1列的矩阵
y_test = model(x_test)
y = y_test.data.numpy()     #tensor转换为numpy

plt.plot(x , y)
plt.plot([0 , 10] , [0.5 , 0.5] , c = 'r')      #做一条垂直于y轴的直线，颜色为red。[0,x][y,y]
                                                #垂直于x轴的直线,[x,x][0,y]
plt.xlabel("Hours")
plt.ylabel("Probability of pass")
plt.grid()
plt.show()
