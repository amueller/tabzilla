#!/bin/bash

# load functions
source ../utils.sh


# this defines DATASETS
source ../HARD_DATASETS_BENCHMARK.sh

name=bench-mothernet-gpu
model=mothernet


# experiment name (will be appended to results files)
experiment_name=$name

# config file
config_file=/home/shared/tabzilla/TabZilla/tabzilla_experiment_config_mothernet_gpu.yml

# results file: check for results here before launching each experiment
result_log=/home/shared/tabzilla/TabZilla/result_log_mothernet.txt

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

  # if the experiment is already in the result log, skip it
  if grep -Fxq "${DATASETS[j]},${model},${experiment_name}" ${result_log}; then
    echo "experiment found in logs. skipping. dataset=${DATASETS[j]}, model=${model}, expt=${experiment_name}"
    continue
  fi


  # args:
  # $1 = model name
  # $2 = dataset name
  # $3 = env name
  # $4 = instance name
  # $5 = experiment name
  echo "MODEL: ${model}"
  echo "ENV: ${env}"
  echo "DATASET: ${DATASETS[j]}"
  echo "EXPERIMENT_NAME: ${experiment_name}"
  #echo "tabzilla_experiment.py --experiment_config ${config_file} --dataset_dir datasets/${DATASETS[j]} --model_name MotherNet "
  python tabzilla_experiment.py --experiment_config ${config_file} --dataset_dir datasets/${DATASETS[j]} --model_name MotherNet
  mv results/default_trial0_results.json results/results_${DATASETS[j]}_mothernet.json

  # >> ${LOG_DIR}/log_${i}_${j}_$(date +"%m%d%y_%H%M%S").txt 2>&1 &                                                                                                   


done

echo "done."