SRCS = accelerator.h common.h main.cpp npu_accelerator.cpp npu_group.cpp software_request_generator.cpp util.cpp util.h util_os.cpp software_layer/*.cpp
MEMSRCS = address_translator.cpp address_translator.h memconfig.cpp memctrl.cpp memctrl.h memctrl_sharedtlb.cpp npumemconfig.cpp ptw.cpp ptw.h spm.cpp spm.h tlb.cpp tlb.h
LINK = -L./DRAMsim3/src/
LIB = -L./DRAMsim3 -ldramsim3
DEBUG = -ggdb -Wall

all: single

single:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	g++ $(LINK) $(DEBUG) $(SRCS) $(MEMSRCS) -o mnpusim -std=c++0x $(LIB)

clean:
	rm -rf mnpusim single_test1/ single_alexnet/

single_test1:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt network_config/netconfig_list/single/test1_network.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt single_test1 misc_config/single.cfg

single_alexnet:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt network_config/netconfig_list/single/alexnet_out_im2col.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt single_alexnet misc_config/single.cfg

single_resnet18:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/resnet18.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt single_resnet18 misc_config/single.cfg

single_vgg11:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/vgg11.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt single_vgg11 misc_config/single.cfg

single_gpt2:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt network_config/netconfig_list/single/gpt2.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt single_gpt2 misc_config/single.cfg

single_deepspeech2:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt network_config/netconfig_list/single/DeepSpeech2_in_im2col.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt single_deepspeech2 misc_config/single.cfg

single_resnet50:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt network_config/netconfig_list/single/Resnet50_in_im2col.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt single_resnet50 misc_config/single.cfg


alexnet_2o0_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/2_blocks/alexnet_2o0_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_2o0_dblock misc_config/single.cfg

alexnet_2o1_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/2_blocks/alexnet_2o1_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_2o1_dblock misc_config/single.cfg

alexnet_4o0_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/4_blocks/alexnet_4o0_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_4o0_dblock misc_config/single.cfg

alexnet_4o1_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/4_blocks/alexnet_4o1_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_4o1_dblock misc_config/single.cfg

alexnet_4o2_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/4_blocks/alexnet_4o2_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_4o2_dblock misc_config/single.cfg

alexnet_4o3_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/4_blocks/alexnet_4o3_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_4o3_dblock misc_config/single.cfg

alexnet_8o0_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/8_blocks/alexnet_8o0_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_8o0_dblock misc_config/single.cfg

alexnet_8o1_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/8_blocks/alexnet_8o1_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_8o1_dblock misc_config/single.cfg

alexnet_8o2_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/8_blocks/alexnet_8o2_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_8o2_dblock misc_config/single.cfg

alexnet_8o3_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/8_blocks/alexnet_8o3_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_8o3_dblock misc_config/single.cfg

alexnet_8o4_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/8_blocks/alexnet_8o4_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_8o4_dblock misc_config/single.cfg

alexnet_8o5_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/8_blocks/alexnet_8o5_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_8o5_dblock misc_config/single.cfg

alexnet_8o6_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/8_blocks/alexnet_8o6_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_8o6_dblock misc_config/single.cfg

alexnet_8o7_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/alexnet_blocked/8_blocks/alexnet_8o7_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt alexnet_8o7_dblock misc_config/single.cfg

resnet50_2o0_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/2_blocks/resnet50_2o0_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_2o0_dblock misc_config/single.cfg

resnet50_2o1_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/2_blocks/resnet50_2o1_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_2o1_dblock misc_config/single.cfg

resnet50_4o0_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/4_blocks/resnet50_4o0_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_4o0_dblock misc_config/single.cfg

resnet50_4o1_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/4_blocks/resnet50_4o1_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_4o1_dblock misc_config/single.cfg

resnet50_4o2_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/4_blocks/resnet50_4o2_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_4o2_dblock misc_config/single.cfg

resnet50_4o3_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/4_blocks/resnet50_4o3_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_4o3_dblock misc_config/single.cfg

resnet50_8o0_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/8_blocks/resnet50_8o0_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_8o0_dblock misc_config/single.cfg

resnet50_8o1_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/8_blocks/resnet50_8o1_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_8o1_dblock misc_config/single.cfg

resnet50_8o2_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/8_blocks/resnet50_8o2_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_8o2_dblock misc_config/single.cfg

resnet50_8o3_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/8_blocks/resnet50_8o3_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_8o3_dblock misc_config/single.cfg

resnet50_8o4_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/8_blocks/resnet50_8o4_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_8o4_dblock misc_config/single.cfg

resnet50_8o5_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/8_blocks/resnet50_8o5_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_8o5_dblock misc_config/single.cfg

resnet50_8o6_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/8_blocks/resnet50_8o6_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_8o6_dblock misc_config/single.cfg

resnet50_8o7_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/resnet50_blocked/8_blocks/resnet50_8o7_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt resnet50_8o7_dblock misc_config/single.cfg

gpt2_2o0_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/gpt2_blocked/2_blocks/gpt2_2o0_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt gpt2_2o0_dblock misc_config/single.cfg

gpt2_2o1_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/gpt2_blocked/2_blocks/gpt2_2o1_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt gpt2_2o1_dblock misc_config/single.cfg

gpt2_4o0_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/gpt2_blocked/4_blocks/gpt2_4o0_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt gpt2_4o0_dblock misc_config/single.cfg

gpt2_4o1_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/gpt2_blocked/4_blocks/gpt2_4o1_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt gpt2_4o1_dblock misc_config/single.cfg

gpt2_4o2_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/gpt2_blocked/4_blocks/gpt2_4o2_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt gpt2_4o2_dblock misc_config/single.cfg

gpt2_4o3_dblock:
	export LD_LIBRARY_PATH=./DRAMsim3:$$LD_LIBRARY_PATH &&\
	./mnpusim arch_config/core_architecture_list/tpu.txt ../custom_network_configs/depth_blocked/gpt2_blocked/4_blocks/gpt2_4o3_dblock.txt dram_config/total_dram_config/single_hbm2_256gbs.cfg npumem_config/npumem_architecture_list/single.txt gpt2_4o3_dblock misc_config/single.cfg
