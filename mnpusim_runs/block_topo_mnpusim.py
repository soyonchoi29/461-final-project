import argparse
import copy
import os
import torch
import torch.nn as nn
from scalesim.topology_utils import topologies


depth_blocked_dir = 'custom_network_configs/depth_blocked'
width_blocked_dir = 'custom_network_configs/width_blocked'
base_dir = 'custom_network_configs'


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
        os.mkdir(depth_blocked_dir + '/' + topo_name)
    os.mkdir(depth_blocked_dir+'/'+topo_name+'/{}_blocks'.format(num_blocks))
    for i in range(len(topos)):
        for layer in blocks[i]:
            topos[i].topo_arrays.append(layer)
        topos[i].topo_load_flag = True
        topos[i].write_topo_file(path=depth_blocked_dir+'/'+topo_name+'/{}_blocks'.format(num_blocks), filename=topo_name+'_block{}.csv'.format(i))


# width-wise blocking of scale-sim topology file
def block_topo_width(topo_file, num_blocks, first_run=False):
    total_topo = read_topology(topo_file)
    topo_name = topo_file[:-4]
    arrs = total_topo.topo_arrays
    blocks = [[] for _ in range(num_blocks)]
    for arr in arrs:
        in_channels_per_block = arr[5] // num_blocks
        out_channels_per_block = arr[6] // num_blocks
        for i in range(num_blocks):
            if i == num_blocks-1:  # last block
                in_num_channels = in_channels_per_block + (arr[5] % num_blocks)
                out_num_channels = out_channels_per_block + (arr[5] % num_blocks)
                to_add = copy.copy(arr)
                to_add[5] = in_num_channels
                to_add[6] = out_num_channels
                blocks[i].append(to_add)
            else:
                to_add = copy.copy(arr)
                to_add[5] = in_channels_per_block
                to_add[6] = out_channels_per_block
                blocks[i].append(to_add)

    topos = [topologies() for _ in range(num_blocks)]
    if first_run:
        os.mkdir(width_blocked_dir + '/' + topo_name)
    os.mkdir(width_blocked_dir+'/'+topo_name+'/{}_blocks'.format(num_blocks))
    for i in range(len(topos)):
        for layer in blocks[i]:
            topos[i].topo_arrays.append(layer)
        topos[i].topo_load_flag = True
        topos[i].write_topo_file(path=width_blocked_dir+'/'+topo_name+'/{}_blocks'.format(num_blocks), filename=topo_name+'_block{}.csv'.format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', metavar='Number of blocks', type=int,
                        default=1,
                        help="Number of blocks to split network into"
                        )

    args = parser.parse_args()
    num_blocks = args.b

    for topo_file in os.listdir(base_dir):
        block_topo_width(topo_file, num_blocks)
