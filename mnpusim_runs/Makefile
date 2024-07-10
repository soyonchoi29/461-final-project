MAIN = ./mNPUsim/
SRCS = ${MAIN}accelerator.h ${MAIN}common.h ${MAIN}main.cpp ${MAIN}npu_accelerator.cpp ${MAIN}npu_group.cpp ${MAIN}software_request_generator.cpp ${MAIN}util.cpp ${MAIN}util.h ${MAIN}util_os.cpp ${MAIN}software_layer/*.cpp
MEMSRCS = ${MAIN}address_translator.cpp ${MAIN}address_translator.h ${MAIN}memconfig.cpp ${MAIN}memctrl.cpp ${MAIN}memctrl.h ${MAIN}memctrl_sharedtlb.cpp ${MAIN}npumemconfig.cpp ${MAIN}ptw.cpp ${MAIN}ptw.h ${MAIN}spm.cpp ${MAIN}spm.h ${MAIN}tlb.cpp ${MAIN}tlb.h
LINK = -L${MAIN}DRAMsim3/src/
LIB = -L${MAIN}DRAMsim3 -ldramsim3
DEBUG = -g -O2

all: single

single:
	export LD_LIBRARY_PATH=${MAIN}DRAMsim3:$$LD_LIBRARY_PATH &&\
	g++ $(LINK) $(DEBUG) $(SRCS) $(MEMSRCS) -o mnpusim -std=c++0x $(LIB)

clean:
	rm -rf mnpusim single_test1/ single_alexnet/

single_test1:
	export LD_LIBRARY_PATH=${MAIN}DRAMsim3:$$LD_LIBRARY_PATH &&\
	${MAIN}mnpusim ${MAIN}arch_config/core_architecture_list/tpu.txt ${MAIN}network_config/netconfig_list/single/test1_network.txt ${MAIN}dram_config/total_dram_config/single_hbm2_256gbs.cfg ${MAIN}npumem_config/npumem_architecture_list/single.txt ./results/single_test1 ${MAIN}misc_config/single.cfg
