import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import inspect
from torch.autograd import Variable


# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# torch.utils.data.DataLoader has __iter__ method, so we can create generator from a DataLoader and
# use .next() method to get the elements
# each element of DataLoader is a tuple of length 2.First is a batch_size*img_shape tensor,second is
# a batch_size*1 tensor. They are input images in a batch and their labels, respectively
# if shuffle is True, we get batches in random order
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

# show some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))
# Module is a class under nn.modules.modules.py, it contains a forward method and a __call__ method,
# in which () is overriden to invoke forward() method.
# Thus, Module is used as template of all neural network classes, as a base class
class Net(nn.Module):
    def __init__(self):
        # here the class Net is implemented from is initialized. The arguments given to this function
        # should match the arguments required by __init__ function in Net's base class
        super(Net, self).__init__()
        # several attributes are initialized with pre-prepared layer objects in nn. Take nn.Conv2d as
        # an example. the parameters 3,6,5 are nb_channels, nb_kernels, and size_kernel. So conv1 is an
        # initialized Conv2d object
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # why is the shape 16*5*5
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        # Conv2d inherits _ConvNd<-Module, so it has implemented __call__ method and will invoke forward()
        # method when using Conv2d with operand (). Attention in the definition of Module, forward method
        # contains only 'raise NotImplementedError', which works like this: The nearest definition of forward
        # method on the chain of inheritance Conv2d<-ConvNd<-Module is invoked. So if it is defined locally
        # in Conv2d(overriding all defined before), it will use lcoal definition. If it doesn't has a local
        # definition but in ConvNd, it will invoke the one defined in ConvNd.Else, it will eventually come
        # to the definition of foward method in Module, which comes to a NotImplementedError
        # Note the difference between _ConvNd(Module) and ConvNd(Function)
        # another thing to mention is that while calling conv1, forward() of conv1 and returns the return of a
        # function nn.functional.conv2d. This function creates f,a nn._functions.ConvNd object, which implements
        # nn.autograd.Function, nn.functional.conv2d returns f(inputs), which is the return of a Function: Variable
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


# implementation: CrossEntropyLoss<-nn._WeightedLoss<-nn._Loss<-Module.So nn.CrossEntropyLoss is callable
# and returns the return of Module.forward(), which is Variable
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
# torch.optim.SGD<-torch.optim.optimizer<-optimizer
# net.parameters(): a generator containing parameter of net
# list(net.parameters()) contains nn.parameter.Parameter<-Variable
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients.In fact it is zeroing the grads of parameters wrapped in optimizer
        optimizer.zero_grad()

        # forward + backward + optimize
        # __call__ function of class Module is implement so as to invoke forward() function
        # that is, () is overriden and Module() objects are callable. This callable returns
        # the return of forward() function, which is a Variable
        outputs = net(inputs)
        # loss is a Variable
        loss = criterion(outputs, labels)
        # backward this Variable. before this, pick an element of params = list(net.parameters()), params[0]
        # params[0].grad are all 0. After backward(), they are assigned with gradient values, but at this
        # stage params[0] remain unchanged.Finally when optimizer.step() is invoked, params[0] is updated
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s'%classes[labels[j]] for j in range(4)))

outputs = net(Variable(images))

# the outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class

# So, let's get the index of the highest energy
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s'% classes[predicted[j][0]] for j in range(4)))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))