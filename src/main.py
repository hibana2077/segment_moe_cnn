import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm

from torch.utils.data import DataLoader

# 設置設備（GPU如果可用，否則使用CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 數據預處理和加載
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = timm.create_model('resnet18').to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 訓練模型
confidence = {}
print("Start training")
epochs = 5
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], loss: {running_loss / len(trainloader):.3f}')

    # EVALUATE
    net.eval()
    confidence_list = []

    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            probs = F.softmax(outputs, dim=1)

            for i in range(len(probs)):
                confidence_list.append(probs[i][labels[i]].item())

    confidence[epoch] = confidence_list

print('Finished Training')

# 在測試集上評估模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

confidence_list = [[] for _ in range(len(confidence[0]))]
for epoch in confidence:
    for i in range(len(confidence[epoch])):
        confidence_list[i].append(confidence[epoch][i])

confidence_list = np.array(confidence_list)
print(confidence_list.shape)

mean = np.mean(confidence_list, axis=1)
std = np.std(confidence_list, axis=1)
variability = std / mean

import matplotlib.pyplot as plt
# plot mean and variability (dot plot)

plt.figure(figsize=(10, 5))
plt.scatter(variability, mean, c='r', marker='x')
plt.xlabel('Variability')
plt.ylabel('Mean')
plt.title('Data map')
plt.savefig('data_map.png')