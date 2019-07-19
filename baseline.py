import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from models import ResNet50
from utils.summary import conv_visualize, mean_std

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)


writer = SummaryWriter(log_dir='old/baseline')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# config = {"mb":0,"mbr":2}
config = {"mb":0,"mbr":0}

net = ResNet50(config)
net = net.to(device)

print(torch.cuda.is_available())

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

step_per_epoch = len(trainloader)
total_epoch = 1000


loss_list = []  
for epoch in range(total_epoch):  # loop over the dataset multiple times

    net.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0),total=step_per_epoch):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        # print statistics
        running_loss += loss.item()

        global_step = epoch*step_per_epoch + i
        if i % 100 == 99:
            writer.add_scalar('train_loss', loss, global_step)
            # conv_visualize(net,writer,global_step)
            mean_std(loss_list,writer,global_step)
            for param_group in optimizer.param_groups:
                writer.add_scalar('lr', param_group['lr'], global_step)


    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(outputs, labels.to(device))
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


        features = net.get_feature(data)   # get feature
        writer.add_embedding(features, metadata=target, global_step= epoch)  # write embedding

    test_loss /= len(testloader.dataset)

    acc = 100. * correct / len(testloader.dataset)

    global_step = epoch*step_per_epoch
    writer.add_scalar('eval_loss', loss, global_step)
    writer.add_scalar('eval_acc', acc, global_step)


print('Finished Training')
writer.close()