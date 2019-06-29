import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

def prepareData():
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                download=True, transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, valid_loader, test_loader, classes, batch_size

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # input is RGB - so 3 layers, output 16 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # input is 16 layers, output 32 3x3 kernels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # input is 32 layers, output 64 3x3 kernels
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.maxpool = nn.MaxPool2d(2,2)
        # input to fc1 - 64*4*4, output - 500
        self.fc1 = nn.Linear(64*4*4, 500)
        # input to fc2 - 500, output - 10
        self.fc2 = nn.Linear(500, 10)

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # conv -> relu -> pool
        x = self.maxpool(F.relu(self.conv1(x)))
        # conv -> relu -> pool
        x = self.maxpool(F.relu(self.conv2(x)))
        # conv -> relu -> pool
        x = self.maxpool(F.relu(self.conv3(x)))
        # flatten for dropout and fc
        x = x.view(-1, 64*4*4)
        # dropout -> fc -> relu
        x = self.dropout(x)
        x = F.relu(self.fc1(x))

        # dropout -> fc
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def setLossAndOptimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return criterion, optimizer

def imshow(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0))) 

train_loader, valid_loader, test_loader, classes, batch_size = prepareData()

model = Model()
print(model)
model.cuda()

criterion, optimizer = setLossAndOptimizer(model)

# training
epochs = 30
valid_loss_min = np.Inf

for i in range(1, epochs+1):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()

    for data, target in train_loader:
        # move to GPU
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # backward pass
        loss.backward()
        optimizer.step()
        # update train loss
        train_loss += loss.item()*data.size(0)
    
    # after every epoch check validation error
    model.eval()
    for data, target in valid_loader:
        data, target = data.cuda(), target.cuda()
        # forward pass and update loss
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)

    print("Epoch: {} \t Training Loss: {:.4f} \t Validation Loss: {:.4f}".format(i, train_loss, valid_loss))
    if valid_loss < valid_loss_min:
        print("Validation loss reduced from {:.4f} to {:.4f}".format(valid_loss_min, valid_loss))
        print("Saving model")
        torch.save(model.state_dict(), 'cifar_cnn.pt')
        valid_loss_min = valid_loss

# load model with best validation loss
model.load_state_dict(torch.load('cifar_cnn.pt'))

# testing
test_loss = 0.0
correct_items_per_class = list(0 for i in range(10))
total_items_per_class = list(0 for i in range(10))

model.eval()
for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    # get predicted class from probabilities
    _, pred = torch.max(output, 1)
    # compare with true labels
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct_numpy = np.squeeze(correct_tensor.cpu().numpy())

    for i in range(batch_size):
        label = target.data[i]
        # update correct items and total items
        correct_items_per_class[label] += correct_numpy[i].item()
        total_items_per_class[label] += 1

test_loss = test_loss/len(test_loader.sampler)

print("Test Loss: {:.4f}".format(test_loss))

# Print accuracies
for i in range(10):
    print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (classes[i],  100 * correct_items_per_class[i] / total_items_per_class[i],
        np.sum(correct_items_per_class[i]), np.sum(total_items_per_class[i])))

print("\n Overall Test Accuracy: %2d%% (%2d/%2d)" % (100 * np.sum(correct_items_per_class) / np.sum(total_items_per_class),
    np.sum(correct_items_per_class), np.sum(total_items_per_class)))

# Plot a few images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

images = images.cuda()

# forward pass
model.eval()
output = model(images)
_, preds_tensor = torch.max(output, 1)
predictions = np.squeeze(preds_tensor.cpu().numpy())

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[predictions[idx]], classes[labels[idx]]),
                 color=("green" if predictions[idx]==labels[idx].item() else "red"))
plt.show()

