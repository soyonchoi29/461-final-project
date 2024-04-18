CFG_PATH=./configs
TOPS_PATH=./topologies
LOG_DIR=./log.txt

for cf in "${CFG_PATH}"/*
do
  for top in "${TOPS_PATH}"/*
  do
    python3 ./scale-sim-v2/scalesim/scale.py -c ${cf} -t ${top} -p ${LOG_DIR}
  done
done
