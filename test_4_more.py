import torch

x_data = [1.0 , 2.0 , 3.0]
y_data = [2.0 , 4.0 , 6.0]

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])

w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

def forward(x):
    return w1 * x ** 2 + w2 * x + b

def loss(x , y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("\t predict (before training)", 4 , forward(4).item())

for epoch in range(100):
    for x , y in zip(x_data , y_data):
        l = loss(x , y)
        l.backward()
        print("\t grad:" , x , y , round(w1.grad.item(),2) , round(w2.grad.item(),2) , round(b.grad.item(),2))

        w1.data -= 0.01 * w1.grad.data
        w2.data -= 0.01 * w2.grad.data
        b.data -= 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print("\t progress:" ,epoch , l.item())

print("predict (after training)", 4 , forward(4).item())