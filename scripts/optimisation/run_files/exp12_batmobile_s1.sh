#!/bin/bash -l
#SBATCH --job-name=E12S1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=25G
## cpus per task
#SBATCH -c 5
#SBATCH --qos=big
##test
#SBATCH --partition=rtx3080
##SBATCH --exclude=szarancza
## cpu, dgxmatinf, rtx2080, rtx3080, dgxa100

#SBATCH --output="E12S1.out"
#SBATCH --error="E12S1.err"

cd $SLURM_SUBMIT_DIR

echo Poczatek `date`

DATA=human
SPLIT=random
FP=krfp
TASK=classification
M1=trees
M2=svm
MR=svm

## no filtering ws
WS_MIN=0.9
  #HI_GAMMA=0.001
HI_METRIC=ratio
  #HI_MIN=-1
## no filtering
UNIMP_MIU=0.005
UNIMP_METRIC=ratio
  #UNIMP_MAX=2

RESULTS_DIR=$SHRD/full_clean_data/2022-12-full-clean-data

for pf_ratio in 0.1 0.2
do
  for seed in 42 43 44 45 46
  do
    for at_once in 1 2 4
    do
      for n_times in 1 2 4
      do
        for hi_gamma in 0.001 0.0001 0.0005
        do
          for hi_min in 0.05 0.15
          do
            for unimp_max in 0.9 0.95
            do     
              singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise-E12.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times $pf_ratio ${WS_MIN} $hi_gamma ${HI_METRIC} $hi_min ${UNIMP_MIU} ${UNIMP_METRIC} $unimp_max $seed ${RESULTS_DIR} optimisation_results/exp12_batmobile/s1_optimise --skip_criterion_check
              
              ## na baseline wpływa tylko pf_ratio i parametry unimportant (filtrowanie cech)
              singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise-E12.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times $pf_ratio ${WS_MIN} $hi_gamma ${HI_METRIC} $hi_min ${UNIMP_MIU} ${UNIMP_METRIC} $unimp_max $seed ${RESULTS_DIR} optimisation_results/exp12_batmobile/s1_baseline --baseline --skip_criterion_check
              
              singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u scripts/optimisation/optimise-E12.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times $pf_ratio ${WS_MIN} $hi_gamma ${HI_METRIC} $hi_min ${UNIMP_MIU} ${UNIMP_METRIC} $unimp_max $seed ${RESULTS_DIR} optimisation_results/exp12_batmobile/s1_optimise_shap --update_shap
            done
          done
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

