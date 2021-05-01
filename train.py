from data_reader import ImageNetDataset, get_transforms
from models import ResNet18
from accuracy import compute_accuracy
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
import time


data_path = '/home/xian/ImageNet'

train_transform = get_transforms(224)
train_dataset = ImageNetDataset(data_path, 'train', train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=6)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

model = ResNet18(1000).to(device)

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

nepochs = 100
for epoch in range(nepochs):
    epoch_start = time.time()
    print('')
    print('Epoch ' + str(epoch))
    for i, data in enumerate(train_dataloader, 0):
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

        if i % 100 == 0:
            acc = compute_accuracy(outputs, labels)
            print('step %i,  loss: %.2e,  accuracy: %.2f' % (i, loss, acc))

    epoch_time = time.time() - epoch_start
    print('Epoch computed in %i s' % int(round(time.time() - epoch_start)))

print('Finished Training')
