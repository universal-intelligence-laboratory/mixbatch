import torch
import torchvision
import torchvision.transforms as transforms
from mixbatch_pytorch import MixBatch_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from models import ResNet50
import torchvision.utils as vutils

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)


def conv_visualize(model,writer,global_step):
    # 可视化卷积核
    # global_step = epoch*step_per_epoch + step
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            in_channels = param.size()[1]
            out_channels = param.size()[0]   # 输出通道，表示卷积核的个数

            k_w, k_h = param.size()[3], param.size()[2]   # 卷积核的尺寸
            kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
            writer.add_image(f'{name}_all', kernel_grid, global_step=global_step) 

writer = SummaryWriter(log_dir='mini_exc/trace16_conv_emb')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.mb = MixBatch_torch(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.mb(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def get_fc2(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.mb(x)
        x = self.fc2(x)
        return x


config = {'mb':0,',mbr':3}
# net = ResNet50(config)
net = Net()
net = net.to(device)

print(torch.cuda.is_available())
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

step_per_epoch = len(trainloader)
total_epoch = 1000

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

        global_step = epoch*step_per_epoch + i
        if i % 100 == 99:
            writer.add_scalar('train_loss', loss, global_step)
            trace = torch.trace(net.fc2.weight).to('cpu').detach().numpy()
            writer.add_scalar('trace', trace, global_step)
            conv_visualize(net,writer,global_step)

    net.eval()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)  # get data
            features = net.get_fc2(data)   # get feature
            writer.add_embedding(features, metadata=target, global_step= global_step)  # write embedding
            break  # no need to write embedding for the whole dataloader

    # net.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in testloader:
    #         data, target = data.to(device), target.to(device)
    #         output = net(data)
    #         test_loss += criterion(outputs, labels.to(device))
    #         pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(testloader.dataset)

    # acc = 100. * correct / len(testloader.dataset)

    # global_step = epoch*step_per_epoch
    # writer.add_scalar('eval_loss', loss, global_step)
    # writer.add_scalar('eval_acc', acc, global_step)


print('Finished Training')
writer.close()
