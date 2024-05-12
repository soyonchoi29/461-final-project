import argparse
import torch
import torch.nn as nn

from torchvision import models, datasets
from torchvision.transforms import ToTensor
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader

dataset = ImageNetV2Dataset("matched-frequency")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
print(dataloader.dataset[0])
exit()


# depth-wise blocking of PyTorch module
def block_model_depth(model, num_blocks):
    pass


# given test data, outputs model accuracy
def test_model(model, test_X, test_y):
    N = test_X.shape[0]
    preds = model(test_X)
    acc = torch.sum(preds == test_y).float() / N
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', metavar='Number of blocks', type=int,
                        default=1,
                        help="Number of blocks to split network into"
                        )

    args = parser.parse_args()
    num_blocks = args.b

    model = getattr(models, 'alexnet')(pretrained=True)
    block_model_depth(model, num_blocks)

