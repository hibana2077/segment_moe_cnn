import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
import timm.optim as tiopt

from torch.utils.data import DataLoader

# 設置設備（GPU如果可用，否則使用CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 數據預處理和加載
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = timm.create_model('resnet18').to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = tiopt.Lookahead(optim.AdamW(net.parameters(), lr=0.0001))

# 訓練模型
confidence = {}
correctness = {}
print("Start training")
epochs = 10
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
    correctness_list = []

    with torch.no_grad():
        temp_outputs = np.array([])
        temp_groundtruth = np.array([])
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            probs = F.softmax(outputs, dim=1)

            # add to temp_outputs and temp_groundtruth
            temp_outputs = np.concatenate((temp_outputs, probs.max(1)[0].cpu().numpy()))
            temp_groundtruth = np.concatenate((temp_groundtruth, labels.cpu().numpy()))

            for i in range(len(probs)):
                confidence_list.append(probs[i][labels[i]].item())

        print(f"Shape of temp_outputs: {temp_outputs.shape}")
        print(f"Shape of temp_groundtruth: {temp_groundtruth.shape}")
        correctness_list = (temp_outputs == temp_groundtruth).astype(int)

    confidence[epoch] = confidence_list
    correctness[epoch] = correctness_list

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
correctness_list = [[] for _ in range(len(correctness[0]))]
for epoch in confidence:
    for i in range(len(confidence[epoch])):
        confidence_list[i].append(confidence[epoch][i])

for epoch in correctness:
    for i in range(len(correctness[epoch])):
        correctness_list[i].append(correctness[epoch][i])

confidence_list = np.array(confidence_list)
correctness_list = np.array(correctness_list)
print(f"Confidence list shape: {confidence_list.shape}")
print(f"Correctness list shape: {correctness_list.shape}")

mean = np.mean(confidence_list, axis=1)
std = np.std(confidence_list, axis=1)
variability = std / mean

# plot mean and variability (dot plot)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(variability, mean, c=correctness_list, cmap='coolwarm')
plt.xlabel('Variability')
plt.ylabel('Mean')
plt.ylim(0, 1)
plt.title('Data map')
plt.savefig('data_map.png')