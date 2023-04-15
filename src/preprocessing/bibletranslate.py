from pipeline import PipeLine
import blocks as B
from loader import BibleLoader

from pathlib import Path
import os
import csv


BIBLE_PATH = "data/bible_EN_SL.json" # change to ccGigaFidafolder
OUT_PATH = "data/bible" # TODO: s config files al neki
BIBLE_PATH = Path(BIBLE_PATH)
OUT_PATH = Path(OUT_PATH)


preprocess_pipeline = PipeLine([
    B.FilterBlock(B.FilterFunctions.regex_remove, '[\n\r\t]'),
    B.FilterBlock(B.FilterFunctions.regex_replace, '[\u00bb\u00ab]', '"'),
    B.FilterBlock(B.FilterFunctions.min_words, 10),
    B.FilterBlock(B.FilterFunctions.max_words, 150)
])

translation_pipeline = PipeLine([
    B.SLONMTTranslateBlock("en", "sl")
])


def main():
    en_loader_path = OUT_PATH / "bible_en_state.json"
    sl_loader_path = OUT_PATH / "bible_sl_state.json"
    dataset_out_path = OUT_PATH / "bible.csv"

    if os.path.exists(en_loader_path) and os.path.exists(sl_loader_path):
        en_loader = BibleLoader.from_checkpoint(en_loader_path)
        sl_loader = BibleLoader.from_checkpoint(sl_loader_path)
    else:
        os.makedirs(OUT_PATH, exist_ok=True)
        en_loader = BibleLoader([str(BIBLE_PATH)], "en", 10)
        sl_loader = BibleLoader([str(BIBLE_PATH)], "sl", 10)
        
    for en_text_list, sl_text_list in zip(en_loader, sl_loader):
        en_dataset_chunk = translation_pipeline(preprocess_pipeline(en_text_list)["outputs"])
        sl_dataset_chunk = preprocess_pipeline(sl_text_list)

        with open(dataset_out_path, "a") as f:
            for input, output in zip(sl_dataset_chunk["outputs"], en_dataset_chunk["outputs"]):
                if input is not None and output is not None:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow([input, output])
        
        en_loader.save_state(en_loader_path)
        sl_loader.save_state(sl_loader_path)


if __name__ == "__main__":
    main()
