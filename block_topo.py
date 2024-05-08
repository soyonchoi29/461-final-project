import argparse
import torch
import torch.nn as nn

from torchvision import models, datasets
from torchvision.transforms import ToTensor
from torchinfo import summary

from scalesim.topology_utils import topologies


blocked_dir = './topologies/blocked'
base_dir = './topologies/custom'

criterion = nn.CrossEntropyLoss()

def read_topology(topology_file):
    topo = topologies()
    topo.load_arrays(topofile=topology_file)
    return topo


# width-wise blocking of scale-sim topology file
def block_topo(topo, num_blocks):
    pass


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

    if __name__ == '__main__':
        model_names = ['alexnet', 'resnet18', 'vgg11']
        # models = [getattr(models, model_names[0])(pretrained=True)]
        models = [getattr(models, model)(pretrained=True) for model in model_names]

        for i in range(len(models)):
            block_model_depth(models[i], criterion)

