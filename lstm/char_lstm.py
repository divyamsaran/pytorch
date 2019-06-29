import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def one_hot_encoder(arr, n_labels):
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

# Split overall sequence into batches - only full mini-batches
# For every batch, send sub-sequence of certain length
# i.e. say first 5 chars of every batch, then next 5 and so on
# Each batch contains N * M chars: 
# N is number of subsequences in a batch
# M is number of subsequences in a batch
# Total number of chars to keep: N * M * K, K = total batches
# Target -> inputs with one char shifted
def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length # batch_size - N, seq_length - M
    total_batches = len(arr)//batch_size_total # K
    
    # Keep only enough characters to make full batches
    arr = arr[:total_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1)) # split into N, M * K
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length): # step size - seq_length
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

class RNNModel(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {char:int_ for int_, char in self.int2char.items()}

        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, len(self.chars))
    
    def forward(self, x, hidden):
        rnn_output, hidden = self.lstm(x, hidden)
        out = self.dropout(rnn_output)

        # Stack up LSTMs
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # initialize hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        return hidden

def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    """
    net -> RNNModel, 
    data -> text data to train model,
    batch_size -> number of mini sequences per mini batch
    seq_length -> sequence length
    lr -> learning rate
    clip -> gradient clipping
    val_frac -> Fraction of data to hold for validation set
    print_every -> print loss every x iterations
    """
    net.train()
    optimizer = torch.optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    net.cuda()

    counter = 0
    n_chars = len(net.chars)
    for i in range(epochs):
        h = net.init_hidden(batch_size)
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            x = one_hot_encoder(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            inputs, targets = inputs.cuda(), targets.cuda()
            h = tuple([h_.data for h_ in h])

            net.zero_grad()

            output, h = net(inputs, h)
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()

            # clip gradients for exploding gradients
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    x = one_hot_encoder(x, n_chars)
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                    # Creating new variables for the hidden state, otherwise we'd backprop through the entire training history
                    val_h = tuple([h_.data for h_ in val_h])

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                    val_losses.append(val_loss.item())
                net.train()
                print("Epoch: {}/{}...".format(i + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[net.char2int[char]]])
        x = one_hot_encoder(x, len(net.chars))
        inputs = torch.from_numpy(x)
        inputs = inputs.cuda()
        
        h = tuple([h_.data for h_ in h])
        out, h = net(inputs, h)

        # For predicting, we use a softmax on the final layer - this gives the probability, 
        # we then pick topk and sample from that
        p = F.softmax(out, dim=1).data
        p = p.cpu() # move to cpu
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h

def sample(net, size, prime='The', top_k=None):
    net.cuda()
    net.eval() # eval mode
    
    # First off, run through the prime characters to build some history
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

with open('data/anna.txt', 'r') as f:
    text = f.read()

# create dictionaries, int->char and char->int
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {char:int_ for int_, char in int2char.items()}

# encode the text based on the dict
encoded = np.array([char2int[ch] for ch in text])

batches = get_batches(encoded, 8, 50)
x, y = next(batches)
# print('x\n', x[:10, :10])
# print('\ny\n', y[:10, :10])

n_hidden = 512
n_layers = 2
net = RNNModel(chars, n_hidden, n_layers)
print(net)

batch_size = 128
seq_length = 100
n_epochs = 20

# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

model_name = 'rnn_20_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)

text = sample(net, 1000, prime='Anna', top_k=5)
with open('char_lstm/prime_anna.txt', 'w') as f:
    f.write(text)
    print("Write successful")

# Later on load from checkpoints
with open('rnn_20_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)
    
loaded = RNNModel(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])
text = sample(loaded, 2000, top_k=5, prime="And Levin said")
with open('char_lstm/prime_and_levin_said.txt', 'w') as f:
    f.write(text)
    print("Write successful")
