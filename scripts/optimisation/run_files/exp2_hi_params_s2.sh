#!/bin/bash -l
#SBATCH --job-name=E2S2-HI
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
## cpus per task
#SBATCH -c 5
#SBATCH --qos=normal
##test
#SBATCH --partition=rtx3080
##SBATCH --exclude=szarancza
## cpu, dgxmatinf, rtx2080, rtx3080, dgxa100

#SBATCH --output="E2S2-HI.out"
#SBATCH --error="E2S2-HI.err"

cd $SLURM_SUBMIT_DIR

echo Poczatek `date`

DATA=human
SPLIT=random
FP=krfp
TASK=classification
M1=svm
M2=trees
MR=svm

## no filtering ws
WS_MIN=-1
##HI_GAMMA=0.001
HI_METRIC=ratio
##HI_MIN=-1
## no filtering
UNIMP_MIU=0.0000001
UNIMP_METRIC=ratio
UNIMP_MAX=2

RESULTS_DIR=$SHRD/full_clean_data/2022-12-full-clean-data

for seed in 42 43 44 45 46
do
  for at_once in 1 2 4
  do
    for n_times in 1 2 4
    do
      for hi_gamma in 0.0001 0.0005 0.001 0.005 0.01 0.05
      do
        for hi_min in 0.01 0.05 0.1 0.15 0.2
        do
          singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} $hi_gamma ${HI_METRIC} $hi_min ${UNIMP_MIU} ${UNIMP_METRIC} ${UNIMP_MAX} $seed ${RESULTS_DIR} optimisation_results/exp2_ws_hi_params/s2_hi_optimise --skip_criterion_check
          singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} $hi_gamma ${HI_METRIC} $hi_min ${UNIMP_MIU} ${UNIMP_METRIC} ${UNIMP_MAX} $seed ${RESULTS_DIR} optimisation_results/exp2_ws_hi_params/s2_hi_baseline --baseline --skip_criterion_check
        done
      done
    done
  done
done


## OPTIONAL ARGUMENTS
## + --baseline
## x --no_contradictive
## + --skip_criterion_check
## x --update_shap
## x --debug

echo Koniec `date`

