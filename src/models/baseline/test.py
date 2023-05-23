from synonyms import SynonymParaphraser

import pandas as pd
import argparse
import os

#TEST_PATH = "../../../data/backtranslate/testset/backtranslate.csv"
#OUT_PATH = "./"
#SYNONYMS_PATH = "../../../../CJVT_Thesaurus-v1.0/CJVT_Thesaurus-v1.0.xml"

def parse_args():
    parser = argparse.ArgumentParser("Baseline paraphraser.")
    parser.add_argument("test", help="Path to test dataset.")
    parser.add_argument("out", help="Path to save output")
    parser.add_argument("synonyms", help="Path to CVJT Thesaurus .xml file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    test_path = args.test
    out_path = args.out
    synonyms_path = args.synonyms

    print("Loading data.")
    data = pd.read_csv(test_path, sep="\t", names=["sentence", "paraphrase"])
    model = SynonymParaphraser(synonyms_path)

    print("Running inference.")
    sentences = list(data["sentence"])
    paraphrases = list(data["paraphrase"])
    outputs = model.generate(sentences)

    print("Saving.")
    df = pd.DataFrame({"sentence": sentences, "paraphrase": paraphrases, "prediction": outputs})
    df.to_csv(os.path.join(out_path, "baseline.csv"), sep="\t")
    