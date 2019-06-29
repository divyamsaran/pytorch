import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

def prepareData():
    batch_size = 20
    num_workers=0
    # download data from https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5baa60a0_flower-photos/flower-photos.zip
    data_dir = 'flower_photos/'
    train_dir = os.path.join(data_dir, 'train/')
    test_dir = os.path.join(data_dir, 'test/')

    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                        transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    test_data = datasets.ImageFolder(test_dir, transform=data_transform)
    

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                            num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                            num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader, classes, batch_size

def setLossAndOptimizer(model_parameters):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_parameters)
    return criterion, optimizer

train_loader, test_loader, classes, batch_size = prepareData()

# Load pretrained vgg16
vgg16 = models.vgg16(pretrained=True)
print(vgg16)

# we will replace the last layer vgg16.classifier[6] by our own fc layer
print(vgg16.classifier[6].in_features) 
print(vgg16.classifier[6].out_features) 

# to not train anything in layers of "features" part of the model, i.e. back prop shouldn't change weight of initial layers
for param in vgg16.features.parameters():
    param.requires_grad = False

n_inputs = vgg16.classifier[6].in_features
custom_layer = nn.Linear(n_inputs, len(classes))

vgg16.classifier[6] = custom_layer

# Load on GPU
vgg16.cuda()

criterion, optimizer = setLossAndOptimizer(vgg16.classifier.parameters())

# training
epochs = 5
for i in range(1, epochs+1):
    train_loss = 0.0

    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = vgg16(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print("Epoch: {} \t Training Loss: {:.4f}".format(i, train_loss/len(train_loader.sampler)))

test_loss = 0.0
correct_items_per_class = list(0 for i in range(len(classes)))
total_items_per_class = list(0 for i in range(len(classes)))

vgg16.eval()

for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    output = vgg16(data)
    loss = criterion(output, target)
    test_loss += loss.item()
    _, pred = torch.max(output, 1)
    # compare with true labels
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct_numpy = np.squeeze(correct_tensor.cpu().numpy())

    for i in range(batch_size):
        label = target.data[i]
        # update correct items and total items
        correct_items_per_class[label] += correct_numpy[i].item()
        total_items_per_class[label] += 1

test_loss = test_loss/len(test_loader.dataset)

print("Test Loss: {:.4f}".format(test_loss))

# Print accuracies
for i in range(len(classes)):
    print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (classes[i],  100 * correct_items_per_class[i] / total_items_per_class[i],
        np.sum(correct_items_per_class[i]), np.sum(total_items_per_class[i])))

print("\n Overall Test Accuracy: %2d%% (%2d/%2d)" % (100 * np.sum(correct_items_per_class) / np.sum(total_items_per_class),
    np.sum(correct_items_per_class), np.sum(total_items_per_class)))

