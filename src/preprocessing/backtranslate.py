from pipeline import PipeLine
import blocks as B
from loader import GIGAFidaLoader

from pathlib import Path
import glob
import os
import csv


GIGAFIDA_PATH = "F:/Python/nlp/ccGigafida" # change to ccGigaFidafolder
OUT_PATH = "data/backtranslate" # TODO: s config files al neki
GIGAFIDA_PATH = Path(GIGAFIDA_PATH)
OUT_PATH = Path(OUT_PATH)


preprocess_pipeline = PipeLine([
    B.FilterBlock(B.FilterFunctions.regex_remove, '[\n\r\t]'),
    B.FilterBlock(B.FilterFunctions.regex_replace, '[\u00bb\u00ab]', '"'), # replace >><< quotes with ""
    B.FilterBlock(B.FilterFunctions.min_words, 30),
    B.FilterBlock(B.FilterFunctions.max_words_cut, 150),
])

backtranslation_pipeline = PipeLine([
    B.SLONMTTranslateBlock("sl", "en"),
    B.SLONMTTranslateBlock("en", "sl")
])


def main():
    loader_path = OUT_PATH / "backtranslate_state.json"
    dataset_out_path = OUT_PATH / "backtranslate.csv"

    if os.path.exists(loader_path):
        loader = GIGAFidaLoader.from_checkpoint(loader_path)
    else:
        os.makedirs(OUT_PATH, exist_ok=True)
        matcher = GIGAFIDA_PATH / "*"
        paths = sorted(glob.glob(str(matcher)))[:100] # remove [:100] for full GIGAFIDA
        loader = GIGAFidaLoader(paths, 10)
    
    print("Starting backtranslation")

    # utf-8 encoding is required due to funny chars
    with open(dataset_out_path, "a", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")

        for text_list in loader:
            print("Backtranslating batch")
            preprocessed = preprocess_pipeline(text_list)
            dataset_chunk = backtranslation_pipeline(preprocessed["outputs"])

            for input, output in zip(dataset_chunk["inputs"], dataset_chunk["outputs"]):
                if output is not None:
                    try:
                        writer.writerow([input, output])
                    except Exception as e:
                        # if writing to csv fails, skip this example
                        print(f"Backtranslate failed  to write to csv: {e}")
                        continue
        
            loader.save_state(loader_path)
            # write to disk after each batch
            f.flush()


if __name__ == "__main__":
    main()
