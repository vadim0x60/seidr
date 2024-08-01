#!/bin/bash
#SBATCH --job-name=seidr            # job name of your choice
#SBATCH --partition=defq          # partition (queue, see info about the queues below)
#SBATCH --nodes=1                   # -N, the number of nodes that will be allocated to this job
#SBATCH --ntasks=1                  # -n, specifies how many instances of your command are executed (you want to launch X independent processes)
#SBATCH --time=05-00:00:00          # time (D-HH:MM:SS)
#SBATCH --output=/home/anastasiia/nl2ml-codex/scripts/logs/%j-stdout.txt    # file to which STDOUT will be written
#SBATCH --error=/home/anastasiia/nl2ml-codex/scripts/logs/%j-stderr.txt    # file to which STDERR will be written
#SBATCH --array=53,61,71,78,79,80,86,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,153,155,163,164,165,166,167,168,169,170,171,173,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,196,197,199,200,201,202,203,204,205,206,211,214,222,225,249,250,251,252,253,254,256,257,258,260,261,265,267,282,283,285,286,287,288,289,295,296%30

module purge
module use /cm/shared/ex3-modules/latest/modulefiles   # Latest ex3-modules
module load slurm/21.08.8                              # To load slurm module

export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

cd ~/nl2ml-codex

# set up API keys and paths
echo $SLURM_ARRAY_TASK_ID
echo ".config_simula"
source .config_simula

# activate environment (with venv, if poetry is not available)
source venv_poetry/bin/activate
poetry shell

# check python, cuda and packages' versions
echo -e "\n\nPython version"
which python3
python3 --version
echo -e "\n\n--Packages--"
python3 -m pip freeze
echo -e "--Packages list finished--\n"


CONFIG="config/psb2/gpt3_5/experiments_psb2_gpt3_5_run_7_offset_711000_no_lexicase_mp100_bf_2_4_16_1_10_100.csv"
OFFSET=711000
TASK_ID=$(( $SLURM_ARRAY_TASK_ID + $OFFSET ))


# -F csv separator, OFS - output separator, if checks task id, $1="" omits task id (not needed in benchmark.py)

# task_id,problem,language,branching_factor,max_programs,drafts_per_prompt,explanations_per_program,repairs_per_explanation,beam_width,log,lexicase_selection,dataset,model_name,run,offset,valid_examples
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
    --experiment_id $RUN"



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
    --experiment_id $RUN

echo "Job finished"