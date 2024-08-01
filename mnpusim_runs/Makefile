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

