import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import imageio
import glob 
import os
from IPython import display

def prepareData():
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 64

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # get the training datasets
    train_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)

    # prepare data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            num_workers=num_workers)
    
    return train_loader, batch_size

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        # fc ->leaky Relu -> dropout
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)

        # fc ->leaky Relu -> dropout
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)

        # fc ->leaky Relu -> dropout
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)

        # fc
        x = self.fc4(x)

        return x

class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = nn.Linear(hidden_dim * 4, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # fc ->leaky Relu -> dropout
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)

        # fc ->leaky Relu -> dropout
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)

        # fc ->leaky Relu -> dropout
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)

        # fc ->tanh
        x = F.tanh(self.fc4(x))

        return x

# For discriminator, we have d_loss = d_real_loss + d_fake_loss
# For discriminator, we want discriminator(real_images) = 1 and discriminator(fake_images) = 0
# For discriminator, to better generalize, we smooth labels to 0.9 instead of 1
# Loss -> sigmoid + binary cross entropy => BCEWithLogitsLoss
# For Generator, we want discriminator(fake_images) = 1, so we flip the labels from 0 to 1
def real_loss(discriminator_out, smooth=False):
    batch_size = discriminator_out.size(0)
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(discriminator_out.squeeze(), labels)
    return loss

def fake_loss(discriminator_out):
    batch_size = discriminator_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(discriminator_out.squeeze(), labels)
    return loss

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

train_loader, batch_size = prepareData()

# Discriminator parameters
input_size = 784 #28*28
d_output_size = 1 # real/fake
d_hidden_size = 32 # last hidden layer

# Generator parameters
z_size = 100 # latent vector
g_output_size = 784
g_hidden_size = 32

discriminator = Discriminator(input_size, d_hidden_size, d_output_size)
generator = Generator(z_size, g_hidden_size, g_output_size)

print(discriminator)
print(generator)

d_optimizer = optim.Adam(discriminator.parameters())
g_optimizer = optim.Adam(generator.parameters())

# Training
num_epochs = 100
print_every = 400

samples = []
losses = []

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

discriminator.train()
generator.train()

for epoch in range(num_epochs):
    for batch_i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        
        real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
        
        d_optimizer.zero_grad()
        # For training discriminator:
        # 1. Compute the discriminator loss on real, training images
        # 2. Generate fake images
        # 3. Compute the discriminator loss on fake, generated images
        # 4. Add up real and fake loss
        # 5. Perform backpropagation + an optimization step to update the discriminator's weights
        
        # 1. Train with real images with smoothing
        D_real = discriminator(real_images)
        d_real_loss = real_loss(D_real, smooth=True)
        
        # 2. Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = generator(z)
        
        # 3. Compute the discriminator losses on fake images        
        D_fake = discriminator(fake_images)
        d_fake_loss = fake_loss(D_fake)
        
        # 4. add up loss
        d_loss = d_real_loss + d_fake_loss

        # 5. Perform backprop
        d_loss.backward()
        d_optimizer.step()
        
        # For training generator
        # 1. Generate fake images
        # 2. Compute the discriminator loss on fake images, using flipped labels
        # 3. Perform backpropagation + an optimization step to update the generator's weights
        g_optimizer.zero_grad()
        
        # 1. Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = generator(z)
        
        # 2. Compute the discriminator loss on fake images, using flipped labels (real loss)
        D_fake = discriminator(fake_images)
        g_loss = real_loss(D_fake) # use real loss to flip labels
        
        # 3. Perform backprop
        g_loss.backward()
        g_optimizer.step()

        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | discriminator loss: {:6.4f} | generator loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item()))
    
    # After each epoch, store losses and generate fake images over a fixed input
    losses.append((d_loss.item(), g_loss.item()))
    
    generator.eval() # eval mode for generating samples
    samples_z = generator(fixed_z)
    samples.append(samples_z)
    generator.train() # back to train mode

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.show()

with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

# Create GIF
for i in range(100):
    view_samples(i, samples)

with imageio.get_writer('gan_pytorch.gif', mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)