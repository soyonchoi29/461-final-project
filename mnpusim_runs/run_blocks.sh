RES_DIR=./results/

# run this script with the single parameter of the name of the run

python3 block_topo_mnpusim.py -b ${2} -t "${1}.csv"
scp ./Makefile ./mNPUsim/Makefile
mkdir "./results/${NAME}_${2}blocks"

for i in $(seq 0 $((${2}-1)))
do
    cd mNPUsim
    NAME="${1}_${2}o${i}_block"
    make "${NAME}"
    cd ..
    scp -r "./mNPUsim/${NAME}" "./results/${1}_${2}blocks/${NAME}"
    echo "block ${i} done!"
done
