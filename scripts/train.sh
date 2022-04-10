#!/bin/bash
#SBATCH --account=def-branzana
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=0-72:00
#SBATCH --cpus-per-gpu=16

# USAGE: sbatch train.sh TASK_NUM NET FOLD

# arg1: task num
# arg2: network [2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres]
# arg3: fold [0, 1, 2, 3, 4]
TASK_NUM=$1
NET=$2
FOLD=$3


# create unique dir for this task
NUM="$(date +"%s")"
TASK_DIR=$SLURM_TMPDIR/$NUM
mkdir $TASK_DIR


# configure task for nnU-Net
export nnUNet_raw_data_base=$TASK_DIR/nnUNet_raw_data_base
export nnUNet_preprocessed=$TASK_DIR/nnUNet_preprocessed
export RESULTS_FOLDER=/project/def-branzana/ebedard/nnUNet_trained_models

mkdir $nnUNet_raw_data_base
mkdir $nnUNet_raw_data_base/nnUNet_raw_data
mkdir $nnUNet_preprocessed


# duplicate environment on local machine
SOURCE_DIR_TEMP=$TASK_DIR/source
mkdir $SOURCE_DIR_TEMP
cd $SOURCE_DIR_TEMP
git clone https://github.com/erikbedard/nnUNet-1.git

VENV_TEMP=$TASK_DIR/venv/nnUNet-temp
virtualenv --no-download $VENV_TEMP
source $VENV_TEMP/bin/activate
pip install --no-index -r $SOURCE_DIR_TEMP/nnUNet-1/requirements-cc.txt
pip install -e $SOURCE_DIR_TEMP/nnUNet-1


# retrieve data path and make local copy
SOURCE_DATA_DIR=/project/def-branzana/ebedard/data
read DATA_ITEM_PATH __ <<< $(find $SOURCE_DATA_DIR -maxdepth 1 -name "*$TASK_NUM*")
DATA_ITEM_BASE=$(basename $DATA_ITEM_PATH)
LOCAL_DATA_DIR=$nnUNet_raw_data_base/nnUNet_raw_data
cp -r $DATA_ITEM_PATH $LOCAL_DATA_DIR


# decrypt if encrypted
if [ $DATA_ITEM_BASE == *.tar.gpg ]; then
cd $LOCAL_DATA_DIR
source /home/ebedard/scripts/decrypt.sh $DATA_ITEM_BASE "OT&BShoulders"
fi


# run nnUNet
NPROC=$(nproc)
nnUNet_plan_and_preprocess -tl $NPROC -tf $NPROC -t $TASK_NUM
nnUNet_train $NET nnUNetTrainerV2 $TASK_NUM $FOLD --npz -c


exit