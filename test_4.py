#反向传播
import torch

x_data = [1.0 , 2.0 , 3.0]
y_data = [2.0 , 4.0 , 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True      #需要计算梯度

def forward(x):
    return x * w

def loss(x , y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

#训练过程
print("predict (before training)" , 4 , forward(4).item())


for epoch in range(100):

    #随机梯度下降
    for x , y in zip(x_data , y_data):
        l = loss(x , y)         #l是一个张量，tensor主要是在建立计算图
        l.backward()    #使用backward后，计算图被释放
        print('\tgrad:' , x , y , w.grad.item())        #item用来把梯度里面的数值拿出来变成python中的标量
        #防止产生计算图
        #取data进行计算不会建立计算图

        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()     #权重里面梯度数据全部清零
    print("progress:", epoch , l.item())
print("predict (after training)" , 4 , forward(4).item())