import argparse
import os
import torch
import torch.nn as nn
from scalesim.topology_utils import topologies


blocked_dir = './topologies/blocked'
base_dir = './topologies/custom'


def read_topology(topology_file):
    topo = topologies()
    topo.load_arrays(topofile=base_dir+'/'+topology_file)
    return topo


# depth-wise blocking of scale-sim topology file
def block_topo_depth(topo_file, num_blocks, first_run=False):
    total_topo = read_topology(topo_file)
    topo_name = topo_file[:-4]
    arrs = total_topo.topo_arrays
    layers_per_block = len(arrs)//num_blocks
    block_idx, layer_idx = 0, 0
    blocks = [[] for _ in range(num_blocks)]
    for arr in arrs:
        if block_idx == num_blocks:
            blocks[block_idx-1].append(arr)
            continue
        if layer_idx < layers_per_block:
            blocks[block_idx].append(arr)
            layer_idx += 1
        else:
            block_idx += 1
            layer_idx = 1
            if block_idx < num_blocks:
                blocks[block_idx].append(arr)
            else:
                blocks[block_idx-1].append(arr)

    topos = [topologies() for _ in range(num_blocks)]
    if first_run:
        os.mkdir(blocked_dir + '/' + topo_name)
    os.mkdir(blocked_dir+'/'+topo_name+'/{}_blocks'.format(num_blocks))
    for i in range(len(topos)):
        for layer in blocks[i]:
            topos[i].topo_arrays.append(layer)
        topos[i].topo_load_flag = True
        topos[i].write_topo_file(path=blocked_dir+'/'+topo_name+'/{}_blocks'.format(num_blocks), filename=topo_name+'_block{}.csv'.format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', metavar='Number of blocks', type=int,
                        default=1,
                        help="Number of blocks to split network into"
                        )

    args = parser.parse_args()
    num_blocks = args.b

    for topo_file in os.listdir(base_dir):
        block_topo_depth(topo_file, num_blocks)
