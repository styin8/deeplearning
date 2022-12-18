import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

"""
 一般图片是H x W x C
  ToTensor(): 把图片转换成 C x H x W, 并把0-255的像素值缩放到0-1
  Normalize: 标准化, 三个通道分别为均值为0.5, 方差为0.5
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

"""
加载数据集, 第一次加载需要把download改成True
DataLoader 可以分批次(batch_size)加载数据集
"""
trianset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)  # 50000张
trainloader = torch.utils.data.DataLoader(trianset, batch_size=36, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)  # 10000张
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)

"""
测试数据集取batch_size=10000
这样可以直接把所有的测试集取出来
"""
test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk')

"""
打印图片
这里的img输入是上面的test_image, 运行时最好把testloader的batch_size改小一点，表示的是显示多少张图片
这个test_image是经过transform之后的图片，所以在打印图片是需要把图片还原成原始图片
因为在标准化的时候对图片做了 (input[channel] - mean[channel]) / std[channel] 的操作 这里的mean=0.5 std=0.5
所以在还原图片时 对图片做 img = img / 2 + 0.5
因为transform之后图片变成了 C x H x W, 还原就要改成 H x W x C ((1, 2, 0)就是对应索引位置 ??)
"""
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# print(' '.join(classes[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))

# 定义网络 损失函数 优化器
net = LeNet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
epochs = 5

for epoch in range(epochs):
    # 用来算累计损失 打印信息时使用
    running_loss = 0.0

    """
    这个trainloader会有1389项, 就是 50000 / 36 
    """
    for step, data in enumerate(trainloader, start=0):
        inputs, labels = data
        # 梯度清零
        optimizer.zero_grad()

        outputs = net(inputs)
        # 计算损失 反向 优化
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # 因为loss是个tensor的形式 所以要用item()取值
        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = net(test_image)
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print(f"epoch: {epoch + 1}, step: {step + 1}, avg loss: {running_loss / 500}, accuracy: {accuracy}")
                running_loss = 0

print("Train Finished!")

# 保存权重
save_path = './lenet.pth'
torch.save(net.state_dict(), save_path)
