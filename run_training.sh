LOG_DIR="./logs/${1}_train_logs"

mkdir ./models
mkdir ${LOG_DIR}
for num_blocks in 1 2 4 8
do
  (python3 "block_${1}.py" -b ${num_blocks} | tee ${LOG_DIR}/${num_blocks}_blocks.txt) || true
done
