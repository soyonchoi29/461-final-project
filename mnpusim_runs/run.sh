RES_DIR=./results/

# run this script with the single parameter of the name of the run

cd mNPUsim
make $1
cd ..
scp ./mNPUsim/${1} ./${1}
echo "${1} done!"
