import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from inferencer import T5Inferencer

import argparse

SEED = 1337
model_size = "small"
best_model_path = "retarT5v2"

inferencer = T5Inferencer(best_model_path, model_size)


def inference(sentence):
    return inferencer.generate([sentence])[0]


def parse_args():
    parser = argparse.ArgumentParser("Paraphrase inference")
    parser.add_argument(
        "--text", required=False, default="Policisti PU Ljubljana so bili v četrtek zvečer okoli 22.30 obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci pristopili do oškodovancev in od njih z nožem v roki zahtevali denar. Ko so jim denar izročili, so storilci s kraja zbežali. Povzročili so za okoli 350 evrov materialne škode.",
        help="Text to paraphrase.",
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(SEED)
 
    sentence = args.text
    print("\nOriginal sentence:")
    print(sentence)
    print("Paraphrased sentence:")
    print(inference(sentence))
