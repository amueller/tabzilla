#!/bin/bash

# load functions
source ../utils.sh


# this defines DATASETS
source ../HARD_DATASETS_BENCHMARK.sh

name=bench-mothernet-gpu-1000
model=mothernet-1000


# experiment name (will be appended to results files)
experiment_name=$name

# config file
config_file=../../TabZilla/tabzilla_experiment_config_mothernet_gpu_subset_1000.yml

# end: EXPERIMENT PARAMETERS
############################

####################
# begin: bookkeeping

# make a log directory
mkdir ${PWD}/logs
LOG_DIR=${PWD}/logs



#################
# run experiments

num_experiments=0

for j in ${!DATASETS[@]};
do

  echo "MODEL: ${model}"
  echo "ENV: ${env}"
  echo "DATASET: ${DATASETS[j]}"
  echo "EXPERIMENT_NAME: ${experiment_name}"
  #echo "tabzilla_experiment.py --experiment_config ${config_file} --dataset_dir datasets/${DATASETS[j]} --model_name MotherNet "
  python ../../TabZilla/tabzilla_experiment.py --experiment_config ${config_file} --dataset_dir ../../TabZilla/datasets/${DATASETS[j]} --model_name MotherNet1000
  mv results/default_trial0_results.json results/results_${DATASETS[j]}_mothernet.json

  # >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &                                                                                                   


done

echo "done."