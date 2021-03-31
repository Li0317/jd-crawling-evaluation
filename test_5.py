#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#可能会报错OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized. #1715
# 若报错则加入上面两行,或者在anaconda prompt中输入conda install nomkl
import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0] , [2.0] , [3.0]])
y_data = torch.Tensor([[2.0] , [4.0] , [6.0]])

class LinearModel(torch.nn.Module):         #把线性模型定义成一个类，所有的模型都继承自module
    def __init__(self):             #__init__为构造函数，必须要有
        super(LinearModel , self).__init__()     #调用父类的__init__
                                                #python3中可以使用直接使用 super().xxx 代替 super(Class, self).xxx
        self.linear = torch.nn.Linear(1 , 1)
        '''torch.nn.Linear是pytorch里的一个类，后面跟()则构造了一个对象           y = wx + b
        这个对象里面包含了权重和偏置这两个tensor：weight和bias
        linear也是继承自nn.Module，可以自动进行反向传播
        class torch.nn.Linear(in_features , out_features , bias = True)
        in_features是输入的维度(几列)   out_features是输出的维度(几列)
        bias为True时有偏置，默认为True
        '''

    def forward(self, x):
        y_pred = self.linear(x)     #在对象后面加()代表实现了一个可调用对象
        return y_pred

model = LinearModel()       #实例化，这个model可以被调用

criterion = torch.nn.MSELoss(reduction='mean')
#criterion = torch.nn.MSELoss(size_average=False)
#构造损失函数,计算损失

''' size_average参数为True表示计算均值，False不计算均值 
    reduce参数为True表示要降维
    MSELoss为(y_pred - y) ** 2
    criterion需要的参数为y_pred和y
    目前关于pytorch版本升级和接口修改，size_average和reduce参数将被弃用，建议将这两个参数去掉，改为reduction='mean'
    关于参数弃用的详细问题点击此网址
https://discuss.pytorch.org/t/userwarning-size-average-and-reduce-args-will-be-deprecated-please-use-reduction-sum-instead/24629
'''
optimizer = torch.optim.SGD(model.parameters() , lr = 0.01)     #优化器设置，不会构造计算图，来自optim
'''torch.optim.SGD是一个类，第一个参数params表示权重，代码中为model.parameters()
model中没有定义相应的权重，即w和b，model里只有linear这个成员，这个成员有两个权重
model父类有个成员函数parameters，即model.parameters()，它会检查model里的所有成员，如果成员里面有相应的权重，
它能把这些全部加到训练的参数集合上

lr参数为学习率
optim下除了SGD还有其他优化器，比如Adam
'''

epoch_list = []
loss_list = []

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred , y_data)

    print("epoch:" , epoch , "loss:" , loss.item())    #这里的Loss会自动调用__str__()，所以不会产生计算图
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    optimizer.zero_grad()       #梯度清零
    loss.backward()     #反向传播
    optimizer.step()    #step()会将optimizer所有参数里面包含的梯度，学习率进行自动更新

print("w = " , model.linear.weight.item())      #需要加item()，因为weight是一个矩阵，这样才能只显示数值
print("b = " , model.linear.bias.item())

x_test = torch.Tensor([[4.0]])      #测试值为1×1的矩阵
y_test = model(x_test)
print("y = " , y_test.data.item())

plt.plot(epoch_list , loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()