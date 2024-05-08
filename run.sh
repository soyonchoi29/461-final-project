CFG_PATH=./google.cfg
TOPS_PATH=./topologies/custom
LOG_DIR=./logs

for top in ${TOPS_PATH}/*
do
  curr_log_dir=${LOG_DIR}/$(basename ${top})
  mkdir ${curr_log_dir}
  python3 ./scale-sim-v2/scalesim/scale.py -c ${CFG_PATH} -t ${top} -p ${curr_log_dir} | tee ${curr_log_dir}/run_log.txt
done
