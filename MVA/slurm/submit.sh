#!/bin/bash
 
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=48g
#SBATCH --time=96:00:00
#SBATCH -C a5000 #|geforce3090
#SBATCH --exclude=gpu2509
#SBATCH -J  GNN
#SBATCH -o  log/run-%j.out  # File to which STDOUT will be written
#SBATCH -e  log/run-%j.out  # File to which STDERR will be written
#SBATCH --mail-type FAIL                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user mondalspandan@gmail.com  # Email to which notifications will be sent

cd /HEP/data/users/smondal5/PocketCoffea/VHccPoCo/MVA
export PATH="/opt/homebrew/bin:/HEP/data/users/smondal5/miniconda3/envs/PocketCoffea/bin:/users/smondal5/miniconda3/condabin:/oscar/runtime/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/oscar/runtime/bin:/users/smondal5/bin:$PATH"
ulimit -s unlimited; ulimit -n 4096
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LR=0.006
python training.py train ../../output_VHcc_02/Saved_columnar_arrays_ZLL/ --signal ZH_Hto2C_Zto2L_2022_preEE --background DYto2L-2Jets_MLL-50_FxFx_2022_preEE --model_type gnn 


