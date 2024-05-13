import argparse
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import models, datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import pil_to_tensor
from torchinfo import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_dir = './models'

# get imagenetv2 data
trans = transforms.Compose([pil_to_tensor,
                            transforms.ConvertImageDtype(dtype=torch.float32),
                            transforms.RandomChoice([transforms.Resize(256),
                                                     transforms.Resize(480)]),
                            transforms.RandomCrop(224),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
dataset = datasets.ImageFolder(root='ImageNetV2-matched-frequency', transform=trans)
dataset = torch.utils.data.Subset(dataset, np.random.randint(int(len(dataset)), size=130))
lengths = [100, 30]
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

model = getattr(models, 'alexnet')(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()


# depth-wise blocked training of PyTorch module
def fit_blocks(model, num_blocks, dataloader):
    start = time.time()

    if num_blocks == 1:
        model = fit(model, dataloader, epochs=5)
        print('training time for {} blocks: %.2f'.format(num_blocks) % (time.time() - start))
        pickle.dump(model, open(models_dir+'/alexnet_{}_block'.format(num_blocks), 'wb'))
        print('saved model with {} blocks!'.format(num_blocks))
        return

    layers = get_children(model)
    layers_per_block = len(layers)//num_blocks
    block_idx, layer_idx = 0, 0
    blocks = [[] for _ in range(num_blocks)]
    for layer in layers:
        if block_idx == num_blocks:
            blocks[block_idx-1].append(layer)
            continue
        if layer_idx < layers_per_block:
            blocks[block_idx].append(layer)
            layer_idx += 1
        else:
            block_idx += 1
            layer_idx = 1
            if block_idx < num_blocks:
                blocks[block_idx].append(layer)
            else:
                blocks[block_idx-1].append(layer)

    for i in range(num_blocks):
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze just the layers in the block i'm currently training
        for layer in blocks[i]:
            layer.requires_grad_(True)
        # now train!
        model = fit(model, dataloader, epochs=5)

    print('training time for {} blocks: %.2f'.format(num_blocks) % (time.time() - start))
    pickle.dump(model, open(models_dir+'/alexnet_{}_blocks'.format(num_blocks), 'wb'))
    print('saved model with {} blocks!'.format(num_blocks))


def fit(model, dataloader, epochs=50):
    model.train()
    for epoch in range(epochs):
        for batch, labels in dataloader:
            x = batch.to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            print('Epoch %d/%d, Avg Loss: %.4f' % (epoch, epochs, loss.item()))
    return model

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children -- model is last child
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


# given test data, outputs model accuracy
def test_model(model, testloader):
    acc = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, labels in testloader:
            x = batch.cuda()
            y = labels.cuda()
            preds = model(x)
            acc += torch.sum(preds.long() == y).float().mean()
            total += 1.

    total_acc = acc/total
    return total_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', metavar='Number of blocks', type=int,
                        default=1,
                        help="Number of blocks to split network into"
                        )

    args = parser.parse_args()
    num_blocks = args.b

    fit_blocks(model, num_blocks, trainloader)
    print('test acc: %.2f' % test_model(model, testloader))

