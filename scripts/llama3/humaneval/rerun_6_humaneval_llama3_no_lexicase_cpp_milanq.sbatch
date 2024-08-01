#!/bin/bash
#SBATCH --job-name=seidr            # job name of your choice
#SBATCH --partition=milanq          # partition (queue, see info about the queues below)
#SBATCH --nodes=1                   # -N, the number of nodes that will be allocated to this job
#SBATCH --ntasks=1                  # -n, specifies how many instances of your command are executed (you want to launch X independent processes)
#SBATCH --time=05-00:00:00          # time (D-HH:MM:SS)
#SBATCH --output=/home/anastasiia/nl2ml-codex/scripts/logs/%j-stdout.txt    # file to which STDOUT will be written
#SBATCH --error=/home/anastasiia/nl2ml-codex/scripts/logs/%j-stderr.txt    # file to which STDERR will be written
#SBATCH --array=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,43,45,46,56,57,58,59,60,61,64,65,66,67,68,69,70,71,72,73,74,75,76,77,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,212,213,214,215,216,217,219,224,226,227,228,239,244,250,254,259,261,263,264,268,274,276,277,278,284,286,287,288,291,292,296,298,299,300,303,306,308,309,319,320,323,327,328,329,330,336,337,340,344,349,360,361,369,370,372,373,376,391,397,399,401,407,408,414,418,428,429,430,431,432,433,434,435,439,440,441,443,444,447,448,450,451,454-984%20

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


CONFIG="config/humaneval/llama3_8b/experiments_humaneval_llama3_8b_run_6_offset_635000_no_lexicase_cpp_mp100_bf_2_4_16_1_10_100.csv"
OFFSET=635000
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



# log what we are running
echo "srun python3 benchmark_humaneval.py \
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
    --experiment_id $RUN \
    --ollama_url $1"


# note that the order of arguments in benchmark.py can be different, so argument names are important
srun python3 benchmark_humaneval.py \
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
    --experiment_id $RUN \
    --ollama_url $1

echo "Job finished"