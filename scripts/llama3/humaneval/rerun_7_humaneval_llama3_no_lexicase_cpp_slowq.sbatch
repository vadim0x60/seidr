#!/bin/bash
#SBATCH --job-name=seidr            # job name of your choice
#SBATCH --partition=slowq          # partition (queue, see info about the queues below)
#SBATCH --nodes=1                   # -N, the number of nodes that will be allocated to this job
#SBATCH --ntasks=1                  # -n, specifies how many instances of your command are executed (you want to launch X independent processes)
#SBATCH --time=05-00:00:00          # time (D-HH:MM:SS)
#SBATCH --output=/home/anastasiia/nl2ml-codex/scripts/logs/%j-stdout.txt    # file to which STDOUT will be written
#SBATCH --error=/home/anastasiia/nl2ml-codex/scripts/logs/%j-stderr.txt    # file to which STDERR will be written
#SBATCH --array=157,318,369,372,373,389,394,395,416,423,433,460,462,472,474,475,476,478,486,487,489,491,493,496,499,500,501,502,503,504,505,506,507,508,509,510,511,512,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,529,530,531,537,538,539,540,541,542,543,544,545,546,547,548,549,550,551,553,554,556,557,566,568,571,572,575,576,578,579,580,583,584,586,589,590,591,592,594,595,599,601,602,605,606,607,608,610,611,612,614,615,620,621,622,623,625,626,628,633,634,644,654,670,671,698,727,757,758,759,760,761,762,763,764,765,766,768,771,772,773,780,783,861,866,870,874,875,881,882,884,887,890,892,893,894,896,900,901,903,907,908,909,918,928,938,941,948,949,950,954,955,957,975,977,983%5

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


CONFIG="config/humaneval/llama3_8b/experiments_humaneval_llama3_8b_run_7_offset_735000_no_lexicase_cpp_mp100_bf_2_4_16_1_10_100.csv"
OFFSET=735000
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