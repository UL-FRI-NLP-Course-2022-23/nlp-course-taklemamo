from preprocessing import Loader, PipeLine
from preprocessing import blocks as B

from pathlib import Path
import glob
import os
import csv

GIGAFIDA_PATH = "" # change to ccGigaFidafolder
OUT_PATH = "" # TODO: s config files al neki

backtranslation_pipeline = PipeLine([
    B.FilterBlock(B.FilterFunctions.min_words, 10),
    B.FilterBlock(B.FilterFunctions.max_words, 100),
    B.SLONMTTranslateBlock("sl", "en"),
    B.SLONMTTranslateBlock("en", "sl")
])

GIGAFIDA_PATH = Path(GIGAFIDA_PATH)
OUT_PATH = Path(OUT_PATH)

def main():
    loader_path = OUT_PATH / "loader_state.json"
    dataset_out_path = OUT_PATH / "datset.csv"

    if os.path.exists(loader_path):
        loader = Loader.from_checkpoint(loader_path)
    else:
        matcher = GIGAFIDA_PATH / "*"
        paths = sorted(glob.glob(str(matcher)))[:100] # remove
        loader = Loader(paths, 10)
    
    for text_list in loader:
        dataset_part = backtranslation_pipeline(text_list)

        print(dataset_part)
        with open(dataset_out_path, "a") as f:
            for input, output in zip(dataset_part["inputs"], dataset_part["targets"]):
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([input, output])
        
        loader.save_state(loader_path)

if __name__ == "__main__":
    main()
