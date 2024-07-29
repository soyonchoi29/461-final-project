RES_DIR=./results/

# run this script with the command line arguments below:
# 1. network name (e.g. alexnet, resnet50)
# 2. number of blocks
# 3. blocking mode ('w' for width or 'd' for depth)


python3 block_topo_mnpusim.py -b ${2} -t "${1}.csv" -m ${3}
scp ./Makefile ./mNPUsim/Makefile
mkdir "./results/${1}_${2}${3}blocks"

for i in $(seq 0 $((${2}-1)))
do
    cd mNPUsim
    if [ ${i} == 0 ]; then
    make clean
    make
    echo "made mnpusim"
    fi

    NAME="${1}_${2}o${i}_${3}block"
    make "${NAME}"
    cd ..
    scp -r "./mNPUsim/${NAME}" "./results/${1}_${2}${3}blocks/${NAME}"
    echo "block ${i} done!"
done
