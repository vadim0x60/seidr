#!/bin/bash
#SBATCH --job-name=seidr            # job name of your choice
#SBATCH --partition=PARTITION          # partition (queue, see info about the queues below)
#SBATCH --nodes=1                   # -N, the number of nodes that will be allocated to this job
#SBATCH --ntasks=1                  # -n, specifies how many instances of your command are executed (you want to launch X independent processes)
#SBATCH --time=05-00:00:00          # time (D-HH:MM:SS)
#SBATCH --output=OUTPUT_DIR/%j-stdout.txt    # file to which STDOUT will be written
#SBATCH --error=OUTPUT_DIR/%j-stderr.txt    # file to which STDERR will be written
#SBATCH --array=1-3%20

module purge
module use MODULES               # Latest ex3-modules
module load SLURM_VERSION        # To load slurm module

export SLURM_CONF=PATH_TO_CONF/slurm.conf

cd ~/nl2ml-codex

# set up API keys and paths
source .config

# activate environment (with venv, if poetry is not available)
source .venv/bin/activate
poetry shell

# check python, cuda and packages' versions
echo -e "\n\nPython version"
which python3
python3 --version
echo -e "\n\n--Packages--"
python3 -m pip freeze
echo -e "--Packages list finished--\n"


CONFIG="config/psb2/llama3_8b/experiments_psb2_llama3_8b_run_1_offset_1310000_lexicase_mp100_bf_custom.csv"
OFFSET=1310000
TASK_ID=$(( $SLURM_ARRAY_TASK_ID + $OFFSET ))


# -F csv separator, OFS - output separator, if checks task id, $1="" omits task id (not needed in benchmark.py)

# task_id,problem,language,branching_factor,max_programs,drafts_per_prompt,explanations_per_program,repairs_per_explanation,beam_width,log,lexicase_selection,dataset,model_name,run,valid_examples,offset
PROBLEM=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $2} }' $CONFIG)
LANG=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $3} }' $CONFIG)
BF=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $4} }' $CONFIG)
MP=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $5} }' $CONFIG)
DRAFTS=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $6} }' $CONFIG)
EXPLANATIONS=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $7} }' $CONFIG)
REPAIRS=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $8} }' $CONFIG)
BW=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $9} }' $CONFIG)
LOG=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $10} }' $CONFIG)
LEXICASE_SELECTION=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $11} }' $CONFIG)
DATASET=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $12} }' $CONFIG)
MODEL_NAME=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $13} }' $CONFIG)
RUN=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $14} }' $CONFIG)
VALID_EXAMPLES=$(awk -F ','  -v ArrayTaskID=$TASK_ID 'BEGIN { OFS=" " } { if ($1==ArrayTaskID) {$1="";print $15} }' $CONFIG)


# log what we are running
echo "srun python3 benchmark.py \
    --task_id $TASK_ID \
    --problem $PROBLEM \
    --language $LANG \
    --branching_factor $BF \
    --max_programs $MP \
    --drafts_per_prompt $DRAFTS \
    --explanations_per_program $EXPLANATIONS \
    --repairs_per_explanation $REPAIRS \
    --beam_width $BW \
    --log $LOG \
    --lexicase_selection $LEXICASE_SELECTION \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --valid_examples $VALID_EXAMPLES \
    --experiment_id $RUN \
    --ollama_url $1"


# note that the order of arguments in benchmark.py can be different, so argument names are important
srun python3 benchmark.py \
    --task_id $TASK_ID \
    --problem $PROBLEM \
    --language $LANG \
    --branching_factor $BF \
    --max_programs $MP \
    --drafts_per_prompt $DRAFTS \
    --explanations_per_program $EXPLANATIONS \
    --repairs_per_explanation $REPAIRS \
    --beam_width $BW \
    --log $LOG \
    --lexicase_selection $LEXICASE_SELECTION \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --valid_examples $VALID_EXAMPLES \
    --experiment_id $RUN \
    --ollama_url $1

echo "Job finished"