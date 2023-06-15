#!/bin/bash -l
#SBATCH --job-name=E3S2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
## cpus per task
#SBATCH -c 5
#SBATCH --qos=normal
##test
#SBATCH --partition=rtx3080
##SBATCH --exclude=szarancza
## cpu, dgxmatinf, rtx2080, rtx3080, dgxa100

#SBATCH --output="E3S2.out"
#SBATCH --error="E3S2.err"

cd $SLURM_SUBMIT_DIR

echo Poczatek `date`

DATA=human
SPLIT=random
FP=krfp
TASK=classification
M1=svm
M2=trees
MR=svm

## no filtering
WS_MIN=-1
HI_GAMMA=0.001
HI_METRIC=ratio
## no filtering
HI_MIN=-1
## no filtering
##UNIMP_MIU=0.0000001
UNIMP_METRIC=ratio
##UNIMP_MAX=2

RESULTS_DIR=$SHRD/full_clean_data/2022-12-full-clean-data

for seed in 42 43 44 45 46
do
  for at_once in 1 2 4
  do
    for n_times in 1 2 4
    do
      for unimp_miu in 0.00001 0.00005 0.0001 0.0005 0.001 0.005
      do
        for unimp_max in 0.7 0.8 0.9 0.95
        do
            singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} ${HI_GAMMA} ${HI_METRIC} ${HI_MIN} $unimp_miu ${UNIMP_METRIC} $unimp_max $seed ${RESULTS_DIR} optimisation_results/exp3_filter_unimportant/s2_optimise --skip_criterion_check
            singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} ${HI_GAMMA} ${HI_METRIC} ${HI_MIN} $unimp_miu ${UNIMP_METRIC} $unimp_max $seed ${RESULTS_DIR} optimisation_results/exp3_filter_unimportant/s2_baseline --baseline --skip_criterion_check
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

