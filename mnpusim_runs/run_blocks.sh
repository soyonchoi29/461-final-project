RES_DIR=./results/

# run this script with the single parameter of the name of the run

scp ./Makefile ./mNPUsim/Makefile
cd mNPUsim
for i in $(seq 0 ${2})
do
    NAME="${1}_${2}o${i}_block"
    make "${NAME}"
    cd ..
    scp -r "./mNPUsim/${NAME}" "./results/${NAME}"
    echo "block ${i} done!"
done
