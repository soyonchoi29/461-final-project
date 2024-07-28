import argparse
import copy
import os
import csv


depth_blocked_dir = 'custom_network_configs/depth_blocked'
width_blocked_dir = 'custom_network_configs/width_blocked'
base_dir = 'custom_network_configs'
og_dir = 'used_networks'


class Layer():
    def __init__(self, name, ifmap_h, ifmap_w, filter_h, filter_w, channels, num_filter, stride, type, ifmap_base_addr, filter_base_addr, ofmap_base_addr):
        self.cfg = {}
        self.cfg['name'] = name
        self.cfg['ifmap_h'] = ifmap_h
        self.cfg['ifmap_w'] = ifmap_w
        self.cfg['filter_h'] = filter_h
        self.cfg['filter_w'] = filter_w
        self.cfg['channels'] = channels
        self.cfg['num_filter'] = num_filter
        self.cfg['stride'] = stride
        self.cfg['type'] = type
        self.cfg['ifmap_base_addr'] = ifmap_base_addr
        self.cfg['filter_base_addr'] = filter_base_addr
        self.cfg['ofmap_base_addr'] = ofmap_base_addr

    def get_row(self):
        return list(self.cfg.values())


class Topo():
    def __init__(self):
        self.name = ''
        self.layers = []


    def read_topology(self, topo):
        self.name = topo[:-4]
        with open(og_dir+'/'+topo, 'r', newline='') as topo_file:
            reader = csv.reader(topo_file)
            next(reader)  # skip header
            for row in reader:
                row = row[:-1]
                for i in range(len(row)):
                    if i != 0 and i != 8:
                        row[i] = int(row[i])
                curr_layer = Layer(*list(row))

                self.layers.append(curr_layer)
        return self


    # depth-wise blocking of scale-sim topology file
    def block_topo_depth(self, topo_file, num_blocks, first_run=False):
        self.read_topology(topo_file)
        layers_per_block = len(self.layers)//num_blocks
        block_idx, layer_idx = 0, 0
        blocks = [[] for _ in range(num_blocks)]
        for layer in self.layers:
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
                    # layer.cfg['filter_base_addr']
                    blocks[block_idx].append(layer)
                else:
                    blocks[block_idx-1].append(layer)

        topos = [Topo() for _ in range(num_blocks)]
        n_blocked_dir = depth_blocked_dir + '/' + self.name + '_blocked'
        if first_run:
            os.mkdir(n_blocked_dir)
        os.mkdir(n_blocked_dir+'/{}_blocks'.format(num_blocks))
        for i in range(len(topos)):
            topos[i].layers = blocks[i]
            topos[i].write_topo_file(path=n_blocked_dir+'/{}_blocks'.format(num_blocks), filename=self.name+'_{}o{}_dblock'.format(num_blocks,i))


    # width-wise blocking of scale-sim topology file
    def block_topo_width(self, topo_file, num_blocks, first_run=False):
        self.read_topology(topo_file)
        blocks = [[] for _ in range(num_blocks)]
        for layer in self.layers:
            in_channels_per_block = layer.cfg['channels'] // num_blocks
            out_channels_per_block = layer.cfg['num_filter'] // num_blocks
            for i in range(num_blocks):  # for each block
                if i == num_blocks-1:  # last block
                    in_num_channels = in_channels_per_block + (layer.cfg['channels'] % num_blocks)
                    out_num_channels = out_channels_per_block + (layer.cfg['num_filter'] % num_blocks)
                    to_add = copy.copy(layer)
                    to_add.cfg['channels'] = in_num_channels
                    to_add.cfg['num_filter'] = out_num_channels
                    blocks[i].append(to_add)
                else:
                    to_add = copy.copy(layer)
                    to_add.cfg['channels'] = in_channels_per_block
                    to_add.cfg['num_filter'] = out_channels_per_block
                    blocks[i].append(to_add)

        topos = [Topo() for _ in range(num_blocks)]
        n_blocked_dir = width_blocked_dir + '/' + self.name + '_blocked'
        if first_run:
            os.mkdir(n_blocked_dir)
        os.mkdir(n_blocked_dir+'/{}_blocks'.format(num_blocks))
        for i in range(len(topos)):
            topos[i].layers = blocks[i]
            topos[i].write_topo_file(path=n_blocked_dir+'/{}_blocks'.format(num_blocks), filename=self.name+'_{}o{}_wblock'.format(num_blocks,i))


    def write_topo_file(self, path, filename):
        # write actual blocked topology file
        with open(path+'/'+filename+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.layers[0].cfg.keys())
            for layer in self.layers:
                writer.writerow(layer.get_row())

        # add path text file (required by mNPUsim)
        file = open(path+'/'+filename+'.txt', 'w')
        file.write('../'+path+'/'+filename+'.csv')
        file.close()

        # add command to makefile
        file = open('Makefile', 'a')
        file.write('\n'+filename+':\n')
        file.write('\texport LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\\\n')
        file.write('\t./mnpusim arch_config/core_architecture_list/tpu.txt ../'+path+'/'+filename+'.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt '+filename+' misc_config/single.cfg\n')
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', metavar='Number of blocks', type=int,
                        default=1,
                        help="Number of blocks to split network into"
                        )
    parser.add_argument('-t', metavar='Topology file', type=str,
                        default='alexnet.csv',
                        help="Name of network topology file to block"
                        )


    args = parser.parse_args()
    num_blocks = args.b
    topo_file = args.t

    topology = Topo()
    topology.block_topo_depth(topo_file, num_blocks, first_run=False)
