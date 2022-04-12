#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --account=def-branzana
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --time=0-71:59         # Choose 3h,12h,24h,72h,7d,28d, or less
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=$1-$2-$3.run
#SBATCH --output=slurm-%j-$1-$2-$3.run

# USAGE: sbatch train.sh TASK_NUM NET FOLD PASSWORD

# arg1: task num (e.g. 600)
# arg2: network [2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres]
# arg3: fold [0, 1, 2, 3, 4]
# arg4: password for decryption (if needed)
TASK_NUM=$1
NET=$2
FOLD=$3
PASSWORD=$4

TASK_START_TIME=$(date +%s)
echo "$(date +"%Y-%m-%d %H:%M:%S"): Starting task..."
echo "Task: $TASK_NUM, Net: $NET, Fold: $FOLD"
nvidia-smi


# create unique dir for this task
NUM="$(date +"%s")"
TASK_DIR=$SLURM_TMPDIR/$NUM
mkdir "$TASK_DIR"


# configure task for nnU-Net
export nnUNet_raw_data_base=$TASK_DIR/nnUNet_raw_data_base
export nnUNet_preprocessed=$TASK_DIR/nnUNet_preprocessed
export RESULTS_FOLDER=/project/def-branzana/ebedard/nnUNet_trained_models

mkdir "$nnUNet_raw_data_base"
mkdir "$nnUNet_raw_data_base"/nnUNet_raw_data
mkdir "$nnUNet_preprocessed"


# duplicate environment on local machine
SOURCE_DIR_TEMP=$TASK_DIR/source
mkdir "$SOURCE_DIR_TEMP"
cd "$SOURCE_DIR_TEMP"
git clone https://github.com/erikbedard/nnUNet-1.git

module load python/3.8
VENV_TEMP=$TASK_DIR/venv/nnUNet-temp
virtualenv --no-download "$VENV_TEMP"
source "$VENV_TEMP"/bin/activate
pip install --no-index -r "$SOURCE_DIR_TEMP"/nnUNet-1/requirements-cc.txt
pip install -e "$SOURCE_DIR_TEMP"/nnUNet-1


# retrieve data path and make local copy
SOURCE_DATA_DIR=/project/def-branzana/ebedard/data
read DATA_ITEM_PATH __ <<< "$(find $SOURCE_DATA_DIR -maxdepth 1 -name "*$TASK_NUM*")"
DATA_ITEM_BASE=$(basename "$DATA_ITEM_PATH")
LOCAL_DATA_DIR=$nnUNet_raw_data_base/nnUNet_raw_data
cp -r "$DATA_ITEM_PATH" "$LOCAL_DATA_DIR"


# decrypt if encrypted
if [[ "$DATA_ITEM_BASE" == *.tar.gpg ]]; then
  cd "$LOCAL_DATA_DIR"
  source "$SOURCE_DIR_TEMP"/nnUNet-1/scripts/decrypt.sh "$DATA_ITEM_BASE" "$PASSWORD"
fi


# run nnUNet
NPROC=$(nproc)
nnUNet_plan_and_preprocess -tl $NPROC -tf $NPROC -t $TASK_NUM

if [[ "$NET" == "3d_cascade_fullres" ]]; then
  TRAINER="3d_cascade_fullres"
else
  TRAINER="nnUNetTrainerV2"
fi
nnUNet_train $NET $TRAINER $TASK_NUM $FOLD --npz -c


TASK_END_TIME=$(date +%s)
echo "$(date +"%Y-%m-%d %H:%M:%S"): Task complete."
HOURS=$(( (TASK_END_TIME - TASK_START_TIME) / 3600 ))
MINS=$(( (TASK_END_TIME - TASK_START_TIME) / 60 ))
echo "Task execution time: $HOURS h $MINS m"

exit