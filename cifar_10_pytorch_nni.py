import torch
import torchvision
import torchvision.transforms as transforms
from mixbatch_pytorch import MixBatch_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from models import ResNet50
import nni

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)


RCV_CONFIG = nni.get_next_parameter()


writer = SummaryWriter(log_dir='res50_fix_exc/NAS_mbat{}'.format(RCV_CONFIG['mb']))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F
net = ResNet50(RCV_CONFIG)
net = net.to(device)

print(torch.cuda.is_available())
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

step_per_epoch = len(trainloader)
total_epoch = 100

best_acc = 0.0
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

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

        if i % 1000 == 999:
            global_step = epoch*step_per_epoch + i
            writer.add_scalar('train_loss', loss, global_step)



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

    test_loss /= len(testloader.dataset)

    acc = 100. * correct / len(testloader.dataset)

    global_step = epoch*step_per_epoch
    writer.add_scalar('eval_loss', loss, global_step)
    writer.add_scalar('eval_acc', acc, global_step)
    nni.report_intermediate_result(acc)
    best_acc = max(best_acc,acc)

nni.report_final_result(best_acc)


print('Finished Training')
writer.close()