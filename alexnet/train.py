import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from model import AlexNet
import os
import time
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
image_path = ''
train_dataset = datasets.ImageFolder(root=image_path + "/train", transform=data_transform['train'])

train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
class_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(class_dict, indent=4)
with open('class_indices.json', 'w') as j:
    j.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

valid_dataset = datasets.ImageFolder(root=image_path + "/val", transform=data_transform['val'])
val_num = len(valid_dataset)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

net = AlexNet(num_classes=5, init_weights=True)
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = 'alexnet.pth'
best_acc = 0.0
epochs = 10

for epoch in range(epochs):
    net.train()
    running_loss = 0
    for step, data in enumerate(train_loader):
        images, labels = data

        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(f"\rtrain loss: {rate * 100:.3f}% [{a}->{b}], {loss:.3f}", end="")
    print()
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data in valid_loader:
            val_images, val_labels = data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        acc_val = acc / val_num
        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(net.state_dict(), save_path)
        print(f"epoch{epoch + 1}: train_loss {running_loss / step}, acc {acc_val}")

print("Train Finished")