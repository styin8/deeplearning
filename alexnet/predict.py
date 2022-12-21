import torch
from model import AlexNet
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img = Image.open('xxx.png')
plt.imshow(img)
img = transform(img)
img = torch.unsqueeze(img, dim=0)

try:
    json_file = open('./class_indices.json', 'r')
    class_indices = json.load(json_file)
    json_file.close()
except Exception as e:
    print(e)
    exit(-1)

net = AlexNet(num_classes=5)
net.load_state_dict(torch.load('alexnet.pth'))

net.eval()
with torch.no_grad():
    # 1.
    # output = net(img)
    # predict = torch.max(output, dim=1)[1].data.numpy()
    # print(class_indices.get(str(int(predict))))
    # 2.
    output = torch.squeeze(net(img))
    predict = torch.softmax(output, dim=0)
    print(predict)
    pre_indices = torch.argmax(predict).numpy()
    print(class_indices[str(pre_indices)], predict[pre_indices].numpy())

plt.show()