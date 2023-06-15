#!/bin/bash -l
#SBATCH --job-name=mock
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
## cpus per task
#SBATCH -c 5
#SBATCH --qos=quick
##test
#SBATCH --partition=rtx3080
##SBATCH --exclude=szarancza
## cpu, dgxmatinf, rtx2080, rtx3080, dgxa100

#SBATCH --output="mock.out"
#SBATCH --error="mock.err"

cd $SLURM_SUBMIT_DIR

echo Poczatek `date`

DATA=human
SPLIT=random
FP=krfp
TASK=classification
M1=trees
M2=svm
MR=svm

WS_MIN=0.7
HI_GAMMA=0.001
HI_METRIC=ratio
HI_MIN=0.1
UNIMP_MIU=0.0001
UNIMP_METRIC=ratio
UNIMP_MAX=0.9

RESULTS_DIR=$SHRD/full_clean_data/2022-12-full-clean-data

for seed in 42 43 44 45 46
do
  for at_once in 1 2 4
  do
    for n_times in 1 2 4 1 2 4
    do
      singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} ${HI_GAMMA} ${HI_METRIC} ${HI_MIN} ${UNIMP_MIU} ${UNIMP_METRIC} ${UNIMP_MAX} $seed ${RESULTS_DIR} mock_repro2
      singularity run --pwd $HOME/projects/metstab_pred -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u optimise.py ${DATA} ${SPLIT} ${FP} ${TASK} ${M1} ${M2} ${MR} $at_once $n_times ${WS_MIN} ${HI_GAMMA} ${HI_METRIC} ${HI_MIN} ${UNIMP_MIU} ${UNIMP_METRIC} ${UNIMP_MAX} $seed ${RESULTS_DIR} baseline_repro2 --baseline
    done
  done
done



# echo A teraz z reliczeniem SHAPów `date`

# for seed in 42 43 44 45 46
# do
#   for at_once in 1 2 4
#   do
#     for n_times in 2 4
#     do
#       singularity run --pwd $HOME/projects/metstab_pred/fast_paper -B $HOME:$HOME,$SHRD:$SHRD $SHRD/metpred3.simg python -u mock_experiment.py $at_once $n_times $seed sdir_todo --update_shap
#     done
#   done
# done


echo Koniec `date`

