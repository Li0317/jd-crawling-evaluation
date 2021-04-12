import torch
batch_size = 1
input_size = 4
hidden_size = 4

idx2char = ['e' , 'h' , 'l' , 'o']
x_data = [1 , 0 , 2 , 2 , 3]
y_data = [3 , 1 , 2 , 3 , 2]

one_hot_lookup = [[1 , 0 , 0 , 0],
                  [0 , 1 , 0 , 0],
                  [0 , 0 , 1 , 0],
                  [0 , 0 , 0 , 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]     #(5 x 4)
#独热向量
inputs = torch.Tensor(x_one_hot).view(-1 , batch_size , input_size) #格式转变，加上batch_size
labels = torch.LongTensor(y_data).view(-1 , 1)      #(seqlen x 1)即(5 x 1)

class Model(torch.nn.Module):
    def __init__(self , input_size , hidden_size , batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size , hidden_size=self.hidden_size)

    def forward(self , input , hidden):
        hidden = self.rnncell(input , hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size , self.hidden_size)

net = Model(input_size , hidden_size , batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters() , lr=0.1)

for epoch in range(20):
    loss = 0
    optimizer.zero_grad()

    hidden = net.init_hidden()      #每一轮初始化h0
    print("Predicted String:" , end='')
    for input , label in zip(inputs , labels):
        hidden = net(input , hidden)
        loss += criterion(hidden , label)
        _ , idx = hidden.max(dim = 1)
        print(idx2char[idx.item()] , end='')

    loss.backward()
    optimizer.step()
    print(" , Epoch [%d / 20] loss = %.4f" % (epoch+1 , loss.item()))