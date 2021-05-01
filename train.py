from data_reader import ImageNetDataset, create_pad_and_resize_transform
from models import ResNet18
from torch import nn, optim
from torch.utils.data import DataLoader
import torch


data_path = '/home/xian/ImageNet'

train_transform = create_pad_and_resize_transform(224)
train_dataset = ImageNetDataset(data_path, 'train', train_transform)
# train_dataset = ImageNetDataset(data_path, 'train', None)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

model = ResNet18(1000).to(device)

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    print('epoch ' + str(epoch))
    for i, data in enumerate(train_dataloader, 0):
        print('i = ' + str(i))
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
