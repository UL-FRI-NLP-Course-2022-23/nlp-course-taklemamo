#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=evaluate.out
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --reservation=fri-vr
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBARCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G

text='Policisti PU Ljubljana so bili v četrtek zvečer okoli 22.30 obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci pristopili do oškodovancev in od njih z nožem v roki zahtevali denar. Ko so jim denar izročili, so storilci s kraja zbežali. Povzročili so za okoli 350 evrov materialne škode.'

srun python ./pipeline.py --model_path /d/hpc/projects/FRI/DL/gs1121/NLP/t5/ParaPlegiq-small --model small --text "$text"