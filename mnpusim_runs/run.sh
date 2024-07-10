RES_DIR=./results/

# run this script with the single parameter of the name of the run

scp ./Makefile ./mNPUsim/Makefile
cd mNPUsim
make $1
cd ..
scp -r ./mNPUsim/${1} ./results/${1}
echo "${1} done!"
