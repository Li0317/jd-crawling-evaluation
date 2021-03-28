import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x * w

#MSE计算(损失值)
def cost(xs , ys):
    cost = 0
    for x , y in zip(xs , ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

#梯度计算
def gradient(xs , ys):
    grad = 0
    for x , y in zip(xs , ys):
        grad += 2 * x * (x * w -y)
    return grad / len(xs)

print("Predict (before training)" , 4 , forward(4))

epoch_list = []
cost_val_list = []
#训练轮数为100
for epoch in range(100):
    cost_val = cost(x_data , y_data)
    grad_val = gradient(x_data , y_data)
    w -= 0.01 * grad_val        #学习率为0.01
    epoch_list.append(epoch)
    cost_val_list.append(cost_val)
    print("Epoch:", epoch , "w = ", w , "Loss = ", cost_val)

print("Predict (after training)" , 4 , forward(4))
plt.plot(epoch_list , cost_val_list)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()