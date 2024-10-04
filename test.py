import argparse
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms
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
dataset = datasets.ImageFolder(root='./data/ImageNetV2-matched-frequency', transform=trans)
dataset = torch.utils.data.Subset(dataset, np.random.randint(int(len(dataset)), size=500))

testloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
# img, label = testloader.dataset[0]

alexnet_1 = pickle.load(open('models/alexnet_1_blocks.pkl','rb'))
# alexnet_2 = pickle.load(open('models/alexnet_2_blocks.pkl','rb'))
# alexnet_4 = pickle.load(open('models/alexnet_4_blocks.pkl','rb'))
# alexnet_8 = pickle.load(open('models/alexnet_8_blocks.pkl','rb'))
#
# resnet_1 = pickle.load(open('models/alexnet_1_blocks.pkl','rb'))
# resnet_2 = pickle.load(open('models/alexnet_2_blocks.pkl','rb'))
# resnet_4 = pickle.load(open('models/alexnet_4_blocks.pkl','rb'))
# resnet_8 = pickle.load(open('models/alexnet_8_blocks.pkl','rb'))

models = [alexnet_1]  # , alexnet_2, alexnet_4, alexnet_8, resnet_1, resnet_2, resnet_4, resnet_8]


# given test data, outputs model accuracy
def test_model(model, testloader):
    if type(model) == tuple:
        model = model[0]  # idk why this turned into a tuple i swear
    total = 0.
    model.eval()
    with torch.no_grad():
        for batch, labels in testloader:
            x = batch.to(device)

            y = labels.to(device)
            print('y: ', y[0])
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            print('preds: ', preds[y])

            corr = preds == y
            print('corr: ', corr)
            total += torch.sum(corr)/100.

    return total/5.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', metavar='Number of blocks', type=int,
                        default=1,
                        help="Number of blocks to split network into"
                        )

    args = parser.parse_args()
    num_blocks = args.b

    for model in models:
         print('test acc: %.2f' % test_model(model, testloader))

    # model = getattr(torchvision.models, 'alexnet')(pretrained=True)
    # print('test acc: %.2f' % test_model(model, testloader))

