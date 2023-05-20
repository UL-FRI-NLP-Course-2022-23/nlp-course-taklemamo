#!/bin/bash
#SBATCH --job-name=infer_t5
#SBATCH --output=infer.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --reservation=fri-vr
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBARCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G

text='Policisti PU Ljubljana so bili v četrtek zvečer okoli 22.30 obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci pristopili do oškodovancev in od njih z nožem v roki zahtevali denar. Ko so jim denar izročili, so storilci s kraja zbežali. Povzročili so za okoli 350 evrov materialne škode.'

srun python ./inference.py --text "$text" --num 3