#!/bin/bash
#SBATCH --job-name=seidr            # job name of your choice
#SBATCH --partition=milanq          # partition (queue, see info about the queues below)
#SBATCH --nodes=1                   # -N, the number of nodes that will be allocated to this job
#SBATCH --ntasks=1                  # -n, specifies how many instances of your command are executed (you want to launch X independent processes)
#SBATCH --time=00-04:00:00          # time (D-HH:MM:SS)
#SBATCH --output=/home/anastasiia/nl2ml-codex/scripts/logs/%j-stdout.txt    # file to which STDOUT will be written
#SBATCH --error=/home/anastasiia/nl2ml-codex/scripts/logs/%j-stderr.txt    # file to which STDERR will be written
#SBATCH --array=0-3

module purge
module use /cm/shared/ex3-modules/latest/modulefiles   # Latest ex3-modules
module load slurm/21.08.8                              # To load slurm module

cd ~/nl2ml-codex

# set up API keys and paths
source .config_0

# activate environment (with venv, if poetry is not available)
source venv/bin/activate


# check python and package versions
echo -e "\n\nPython version"
which python3
python3 --version
echo -e "\n\nPackages"
pip freeze


CONFIG="config/experiments_mp100_bf2_4_16.csv"


# -F csv separator, OFS - output separator, if checks task id, $1="" omits task id (not needed in benchmark.py)
ARGS=$(awk -F ','  -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $0} }' $CONFIG)

# problem,language,branching_factor,max_programs,beam_width,debug_prompt_id,log
PROBLEM=$(awk -F ','  -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $2} }' $CONFIG)
LANG=$(awk -F ','  -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $3} }' $CONFIG)
BF=$(awk -F ','  -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $4} }' $CONFIG)
MP=$(awk -F ','  -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $5} }' $CONFIG)
BW=$(awk -F ','  -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $6} }' $CONFIG)
PROMPT=$(awk -F ','  -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $7} }' $CONFIG)
LOG=$(awk -F ','  -v ArrayTaskID=$SLURM_ARRAY_TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $8} }' $CONFIG)

echo "srun python3 run_experiments.py ${ARGS}"


# note that the order of arguments in benchmark.py can be different, so argument names are important
srun python3 benchmark.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --problem $PROBLEM \
    --language $LANG \
    --branching_factor $BF \
    --max_programs $MP \
    --beam_width $BW \
    --debug_prompt_id $PROMPT \
    --log $LOG

echo "Job finished"