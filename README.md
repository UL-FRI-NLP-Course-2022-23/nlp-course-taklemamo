# Natural language processing course 2022/23: Sentence Paraphrasing 

Team members:
 * `Blaž Rolih`, `63190255`, `br9136@student.uni-lj.si`
 * `Grega Šuštar`, `63180294`, `gs1121@student.uni-lj.si`
 * `ŽIGA ROT`, `63220470`, `zr13891@student.uni-lj.si`
 
Group public acronym/name: taklemamo
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

The report can be found in folder `report`.

## How to setup the environment
---

### 1. Create new conda environment (optional)
```
# Setup new conda environment
conda create --name taklemamo python=3.8

# Activate environment
conda activate taklemamo
```

### 2. Clone GIT repository
```
# Clone repository with HTTP
git clone https://github.com/UL-FRI-NLP-Course-2022-23/nlp-course-taklemamo.git
# or SHH
git clone git@github.com:UL-FRI-NLP-Course-2022-23/nlp-course-taklemamo.git

# move into the directory
cd nlp-course-taklemamo/
```

### 3. Install required packages
```
# Manually install pytorch for CUDA if you don't have it already
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

# Manually install classla
pip install classla

# Install required packages
pip install -r requirements.txt
```

## How to run the models
---
We have Fine-tuned two T5 models for paraphrasing sentences in slovene. Both models can be accessed thorugh the HuggingFace Hub.
- https://huggingface.co/GregaSustar/ParaPlegiq-small
- https://huggingface.co/GregaSustar/ParaPlegiq-large

***We Strongly recommend running the models on a Cluster as they Require a lot of VRAM.*** 

***If you are running the models on a cluster we recommend that you first empty your '.cache/huggingface/hub' folder***

---
```
# Move into the evaluation directory
cd src/evaluation/
```

To run the models you can use the ```pipeline.py``` script. The script accepts the following arguments:

```--model model_name``` - Specify which model to use (small, large or baseline). default = small.

```--model_path  dir_path``` - Specify the location of the model. **ONLY IF YOU ALREDY HAVE IT DOWNLOADED LOCALLY**. If not specified the model will be downloaded from HuggingFace Hub and stored in the .cache folder.

```--thesaurus_path  file_path``` - If you running the baseline model you have to specify the path to the *'CJVT_Thesaurus-v1.0.xml'*. default = ./CJVT_Thesaurus-v1.0.xml

```--run_on_testset``` - If you want to evaluate the model on the entire testset. default = False

```--text "sentence"``` - If you want to evaluate the model on a single sentence.

#### Examples:
```
# Write the paragraph you want to paraphrase and store it in a variable

text='Policisti PU Ljubljana so bili v četrtek zvečer okoli 22.30 obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci pristopili do oškodovancev in od njih z nožem v roki zahtevali denar. Ko so jim denar izročili, so storilci s kraja zbežali. Povzročili so za okoli 350 evrov materialne škode.'

# Run small model on a single sentence
python ./pipeline.py --model small --text "$text"

# Run large model on the entire test set
python ./pipeline.py --model large --run_on_testset
```

If you want to run the **baseline** model you first have to download the CJVT_Thesaurus-v1.0.xml file from clarin.si
```
# Download the zip file
curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1166{/CJVT_Thesaurus-v1.0.zip}

# install unzip (optional)
sudo apt install unzip

# Unzip
unzip CJVT_Thesaurus-v1.0.zip
```

Run the baseline model
```
# Run the baseline model on the entire test set
python ./pipeline.py --model baseline --thesaurus_path ./CJVT_Thesaurus-v1.0.xml --run_on_testset
```

To run the models on the Arnes cluster we have also prepared a simple script ```run_pipeline.sh```

```
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

srun python ./pipeline.py --model small --text "$text"
```

You can adjust the script as described above to run whatever model you like. To run the script you can use the following command
```
# Run SBATCH script
sbatch run_pipeline.sh

# Check if job is running 
squeue --me

# The ouput will be redirected to evaluate.out
```

