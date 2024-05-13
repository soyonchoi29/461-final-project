LOG_DIR=./logs/alexnet_train_logs

mkdir ./models
mkdir ${LOG_DIR}
for num_blocks in 1 2 4 8
do
  (python3 block_alexnet.py -b ${num_blocks} | tee ${LOG_DIR}/${num_blocks}_blocks.txt) || true
done
