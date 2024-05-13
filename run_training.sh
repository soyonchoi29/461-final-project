LOG_DIR=./logs/alex_train_logs

mkdir ./models
for num_blocks in 1 2 4 8
do
  python3 block_alexnet.py -b ${num_blocks} | tee ./logs/alex_train_logs/${num_blocks}_blocks.txt || true
done
