CFG_PATH=./google.cfg
TOPS_PATH=./topologies/blocked
LOG_DIR=./logs

for base in ${TOPS_PATH}/*
do
  mkdir ${LOG_DIR}/$(basename ${base})
  for blocked_tops in ${base}/*
  do
    curr_block_log_dir=${LOG_DIR}/$(basename ${base})/$(basename ${blocked_tops})
    mkdir ${curr_block_log_dir}
    for block in ${TOPS_PATH}/$(basename ${base})
    do
      python3 ./scale-sim-v2/scalesim/scale.py -c ${CFG_PATH} -t ${block} -p ${curr_block_log_dir} | tee ${curr_block_log_dir}/run_log.txt
    done
  done
done
