from PIL import Image
import torch
import torchvision.transforms as transforms
from model import LeNet

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk')

net = LeNet()
# 加载权重
net.load_state_dict(torch.load('lenet.pth'))

im = Image.open('img.png')
im = transform(im)
# 上面的im只有三个通道 C x H x W, 而预测时需要有四个通道 B x C x H x W, 所以需要在第 0 维度加一个通道
im = torch.unsqueeze(im, dim=0)

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
print(classes[int(predict)])