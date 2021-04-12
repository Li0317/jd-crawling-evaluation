import torch
batch_size = 1
input_size = 4
hidden_size = 4
num_layers = 1
seq_len = 5

idx2char = ['e' , 'h' , 'l' , 'o']
x_data = [1 , 0 , 2 , 2 , 3]
y_data = [3 , 1 , 2 , 3 , 2]

one_hot_lookup = [[1 , 0 , 0 , 0],
                  [0 , 1 , 0 , 0],
                  [0 , 0 , 1 , 0],
                  [0 , 0 , 0 , 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(seq_len , batch_size , input_size)
labels = torch.LongTensor(y_data)

class Model(torch.nn.Module):
    def __init__(self , input_size , hidden_size , batch_size , num_layers = 1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size        #为了构造h0,即下方forward中hidden的设置
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size , hidden_size=self.hidden_size ,
                                num_layers=num_layers)

    def forward(self , input):
        hidden = torch.zeros(self.num_layers , self.batch_size , self.hidden_size)  #shape为(num_layers , batch , hidden_size)
        out , _ = self.rnn(input , hidden)
        #input.shape(seqsize , batch , input_size)
        #hidden.shape(numlayers , batch , hidden_size)
        return out.view(-1 , self.hidden_size)

net = Model(input_size , hidden_size , batch_size , num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters() , lr=0.1)

for epoch in range(20):
    optimizer.zero_grad()
    outputs = net(inputs)       #inputs为(seq , batch , input_size)
    loss = criterion(outputs , labels)  #labels为(seq , batch ,  1)      outputs为(seq , batch , hidden_size)
    loss.backward()
    optimizer.step()

    _ , idx = outputs.max(dim = 1)
    idx = idx.data.numpy()
    print('Predicted:',''.join([idx2char[x] for x in idx]) , end='')
    print(', Epoch [%d / 20] loss = %.4f' % (epoch+1 , loss.item()))