#!/bin/bash

# small test script to make sure that TabSurvey code and environments work.

cd /home/shared/tabzilla/TabSurvey

ENV_NAME=torch

# name of the model/algorithm
MODEL_NAME=DeepFM

# all datasets should be in this folder. the dataset folder should be in ${DATASET_BASE_DIR}/<dataset-name>
DATASET_BASE_DIR=./datasets

##########################################################
# define lists of datasets and models to evaluate them on

# MODELS_ENVS=(
#   "LinearModel:$SKLEARN_ENV"
#   "KNN:$SKLEARN_ENV"
#   "DecisionTree:$SKLEARN_ENV"
#   )

CONFIG_FILE=tabzilla_experiment_config.yml


# DATASETS=(
#   openml__california__361089
# )

# conda init bash
eval "$(conda shell.bash hook)"

for dataset in ./datasets/*; do
    printf "\n|----------------------------------------------------------------------------\n"
    printf "| starting dataset ${dataset}\n"
    printf '|| Training %s on dataset %s in env %s\n\n' "$model" "$dataset" "$env"

    conda activate ${ENV_NAME}

    dataset_dir=${dataset}
    python tabzilla_experiment.py --experiment_config ${CONFIG_FILE} --dataset_dir ${dataset_dir} --model_name ${MODEL_NAME}

    # zip results into a new directory, and remove unzipped results
    zip -r results_${dataset}_${model}.zip ./results
    rm -r ./results

    conda deactivate
done
