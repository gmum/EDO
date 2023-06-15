#!/bin/bash -l
#SBATCH --job-name=E5S1-42
#SBATCH --ntasks-per-node=1
#SBATCH --mem=25G
## cpus per task
#SBATCH -c 5
#SBATCH --qos=quick
##test
#SBATCH --partition=rtx3080
##SBATCH --exclude=szarancza
## cpu, dgxmatinf, rtx2080, rtx3080, dgxa100

#SBATCH --output="E5S1-42.out"
#SBATCH --error="E5S1-42.err"

cd $SLURM_SUBMIT_DIR

echo Poczatek `date`

DATA=human
SPLIT=random
FP=krfp
TASK=classification
M1=trees
M2=svm
MR=svm

## no filtering
WS_MIN=-1
HI_GAMMA=0.001
HI_METRIC=ratio
## no filtering
HI_MIN=-1
## no filtering
UNIMP_MIU=0.0000001
UNIMP_METRIC=ratio
UNIMP_MAX=2

RESULTS_DIR=$SHRD/full_clean_data/2022-12-full-clean-data

for seed in 42
do
  for at_once in 1 2 4
  do
    for n_times in 2 4 1
    do
      singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} ${HI_GAMMA} ${HI_METRIC} ${HI_MIN} ${UNIMP_MIU} ${UNIMP_METRIC} ${UNIMP_MAX} $seed ${RESULTS_DIR} optimisation_results/exp5_shap/s1_optimise_cc --update_shap
      # singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} ${HI_GAMMA} ${HI_METRIC} ${HI_MIN} ${UNIMP_MIU} ${UNIMP_METRIC} ${UNIMP_MAX} $seed ${RESULTS_DIR} optimisation_results/exp5_shap/s1_baseline_cc --baseline --update_shap
      singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} ${HI_GAMMA} ${HI_METRIC} ${HI_MIN} ${UNIMP_MIU} ${UNIMP_METRIC} ${UNIMP_MAX} $seed ${RESULTS_DIR} optimisation_results/exp5_shap/s1_optimise_nocc --skip_criterion_check --update_shap
      # singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} ${HI_GAMMA} ${HI_METRIC} ${HI_MIN} ${UNIMP_MIU} ${UNIMP_METRIC} ${UNIMP_MAX} $seed ${RESULTS_DIR} optimisation_results/exp5_shap/s1_baseline_nocc --baseline --skip_criterion_check --update_shap
    done
  done
done


## OPTIONAL ARGUMENTS
## x --baseline
## x --no_contradictive
## + --skip_criterion_check
## + --update_shap
## x --debug

echo Koniec `date`

