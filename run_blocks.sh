CFG_PATH=./configs/scale.cfg
TOPS_PATH=./topologies/width_blocked
LOG_DIR=./logs/width

for base in ${TOPS_PATH}/*
do
  model_log_dir=${LOG_DIR}/$(basename ${base})
  mkdir ${model_log_dir}
  for blocks in ${base}/*
  do
    blocks_log_dir=${model_log_dir}/$(basename ${blocks})
    mkdir ${blocks_log_dir}
    for block in ${blocks}/*
    do
      curr_block_log_dir=${blocks_log_dir}/$(basename ${block})
      mkdir ${curr_block_log_dir}
      python3 ./scale-sim-v2/scalesim/scale.py -c ${CFG_PATH} -t ${block} -p ${curr_block_log_dir} | tee ${curr_block_log_dir}/run_log.txt
    done
  done
done
