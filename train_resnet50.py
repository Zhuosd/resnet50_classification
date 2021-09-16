# D:\picture\jian7_val

# 瀵煎叆鐩稿簲鐨勫簱
import torch
import time
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# 鏋勫缓Resnet妯″瀷
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# 瀹氫箟璁粌鍑芥暟
def train_and_valid(model, loss_function, optimizer, epochs=300):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)#,device_ids = [0, 1, 2,3,4,5,6,7]
    model.to(device)



    ###############
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3,4,5,6,7]).cuda()

    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            # inputs = inputs.to(device)
            inputs = inputs.cuda()
            # labels = labels.to(device)
            labels = labels.cuda()

            # 鍥犱负杩欓噷姊害鏄疮鍔犵殑锛屾墍浠ユ瘡娆¤寰楁竻闆?
            optimizer.zero_grad()

            outputs = model(inputs)
            # print('labels',labels)
            # print('outputs',outputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in enumerate(test_loader):
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                inputs = inputs.cuda()

                labels = labels.cuda()

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        # 灏嗘瘡涓€杞殑鎹熷け鍊煎拰鍑嗙‘鐜囪褰曚笅鏉?
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc

            best_epoch = epoch + 1
            torch.save(model, 'history_best_res.pt')

        epoch_end = time.time()
        # 鎵撳嵃姣忎竴杞殑鎹熷け鍊煎拰鍑嗙‘鐜囷紝鏁堟灉鏈€浣崇殑楠岃瘉闆嗗噯纭巼
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        torch.save(model, 'history_res.pt')

    return history


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    # 鍒濆鍖栬缃?
    BATCH_SIZE = 8
    num_epochs = 300
    num_classes=5
    # 璁剧疆璁粌闆?
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 闅忔満瑁佸壀鍒?24x224澶у皬
        transforms.RandomHorizontalFlip(),  # 闅忔満姘村钩缈昏浆
        transforms.RandomRotation(degrees=15),  # 闅忔満鏃嬭浆
        transforms.ToTensor(),  # 杞寲鎴怲ensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 姝ｅ垯鍖?
    ])
    train_dataset = ImageFolder("./train", transform=train_transform)  # 璁粌闆嗘暟鎹?
    # train_dataset = ImageFolder("/home/zhubin/AAA_zky/picture/jian7_tra", transform=train_transform)  # 璁粌闆嗘暟鎹?
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=2)  # 鍔犺浇鏁版嵁

    # 璁剧疆娴嬭瘯闆?
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize鍒?24x224澶у皬
        transforms.ToTensor(),  # 杞寲鎴怲ensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 姝ｅ垯鍖?
    ])
    test_dataset = ImageFolder("./validation", transform=test_transform)  # 娴嬭瘯闆嗘暟鎹?
    # test_dataset = ImageFolder("/home/zhubin/AAA_zky/picture/jian7_val", transform=test_transform)  # 娴嬭瘯闆嗘暟鎹?
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=2)  # 鍔犺浇鏁版嵁

    # 鎵撳嵃璁粌闆嗘祴璇曢泦澶у皬
    train_data_size = len(train_dataset)
    valid_data_size = len(test_dataset)
    print(train_data_size, valid_data_size)

    # net = torch.nn.DataParallel(ResNet50())

    net = ResNet50(num_classes=num_classes).cuda()  # 璁剧疆涓篏PU璁粌 #璁板緱鏀规垚GPU

    loss_function = nn.CrossEntropyLoss()  # 璁剧疆鎹熷け鍑芥暟
    optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 璁剧疆浼樺寲鍣ㄥ拰瀛︿範鐜?
    # 寮€濮嬭缁?
    history = train_and_valid(net, loss_function, optimizer, num_epochs)

    torch.save(net, 'history_res.pt')
    # 灏嗚缁冨弬鏁扮敤鍥捐〃绀哄嚭鏉?
    # history = np.array(history)
    # plt.plot(history[:, 0:2])
    # plt.legend(['Tr Loss', 'Val Loss'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Loss')
    # plt.ylim(0, 1.1)
    # # plt.savefig(dataset+'_loss_curve.png')
    # plt.show()
    #
    # plt.plot(history[:, 2:4])
    # plt.legend(['Tr Accuracy', 'Val Accuracy'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Accuracy')
    # plt.ylim(0, 1.1)
    # # plt.savefig(dataset+'_accuracy_curve.png')
    # plt.show()

