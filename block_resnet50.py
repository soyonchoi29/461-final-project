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

from torchvision.models import resnet50, ResNet50_Weights

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
dataset = datasets.ImageFolder(root='./data/ImageNetV2-matched-frequency', transform=trans)
dataset = torch.utils.data.Subset(dataset, np.random.randint(int(len(dataset)), size=130))
lengths = [100, 30]
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

# Using pretrained weights:
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
criterion = nn.CrossEntropyLoss()


# depth-wise blocked training of PyTorch module
def fit_blocks(model, num_blocks, dataloader, epochs=50):
    start = time.time()

    if num_blocks == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        model = fit(model, optimizer, dataloader, epochs=epochs)
    else:
        layers = get_children(model)
        layers_per_block = len(layers)//num_blocks
        epochs_per_block = epochs//num_blocks
        block_idx, layer_idx = 0, 0
        blocks = [[] for _ in range(num_blocks)]
        rem_epochs = 0

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
            print('\n about to train block {}!'.format(i))
            # freeze all layers
            params = []
            epochs_block = epochs_per_block
            for param in model.parameters():
                param.requires_grad = False
            # unfreeze just the layers in the block i'm currently training
            for layer in blocks[i]:
                for param in layer.parameters():
                    param.requires_grad = True
                    params.append(param)
            if params:
                if i == num_blocks-1:
                    epochs_block = epochs_per_block + (epochs % num_blocks)
                if rem_epochs is not None:
                    epochs_block = epochs_per_block + rem_epochs
                optimizer = torch.optim.Adam(params, lr=5e-5)
                # now train!
                print('going to train for {} epochs!'.format(epochs_block))
                model, this_rem_epochs = fit(model, optimizer, dataloader, epochs=epochs_block)
                if this_rem_epochs is not None:
                    rem_epochs += this_rem_epochs

    print('training time for {} blocks: %.2f'.format(num_blocks) % (time.time() - start))
    pickle.dump(model, open(models_dir+'/resnet50_{}_blocks'.format(num_blocks), 'wb'))
    print('saved model with {} blocks!'.format(num_blocks))


def fit(model, optimizer, dataloader, epochs=50, conv_tol=0.03):
    model.train()
    losses = []
    prev_loss = -100
    rem_epochs = epochs
    for epoch in range(epochs):
        for batch, labels in dataloader:
            x = batch.to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        with torch.no_grad():
            print('Epoch %d/%d, Avg Loss: %.4f' % (epoch, epochs, loss.item()))

            if conv_tol > abs(prev_loss-loss):
                rem_epochs -= epoch
                return model, rem_epochs

            prev_loss = loss

    print(losses)
    return model, None

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
    total_loss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, labels in testloader:
            x = batch.to(device)
            y = labels.to(device)
            preds = model(x)
            # print('y=',y)
            # print('preds=',preds)
            loss = criterion(preds, y)
            total_loss += loss
            total += 1

    total_loss = total_loss/total
    return total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', metavar='Number of blocks', type=int,
                        default=1,
                        help="Number of blocks to split network into"
                        )

    args = parser.parse_args()
    num_blocks = args.b

    fit_blocks(model, num_blocks, trainloader)
    print('test loss: %.2f' % test_model(model, testloader))

