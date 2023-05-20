import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

import argparse


model_size = "large"
SEED = 1337

best_model_path = "/d/hpc/projects/FRI/DL/gs1121/NLP/t5/checkpoint-24688/"
model = T5ForConditionalGeneration.from_pretrained(best_model_path)
tokenizer = T5Tokenizer.from_pretrained(f'cjvt/t5-sl-{model_size}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)


def inference(sentence, num_paraphrases):
    # text = "paraphrase: " + sentence
    text = "parafraziraj: " + sentence
    max_len = 512

    encoding = tokenizer.encode_plus(text, max_length=max_len, padding="max_length", return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=max_len,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=num_paraphrases
    )

    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    return final_outputs


def parse_args():
    parser = argparse.ArgumentParser("Paraphrase inference")
    parser.add_argument(
        "--text", required=False, default="Policisti PU Ljubljana so bili v četrtek zvečer okoli 22.30 obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci pristopili do oškodovancev in od njih z nožem v roki zahtevali denar. Ko so jim denar izročili, so storilci s kraja zbežali. Povzročili so za okoli 350 evrov materialne škode.",
        help="Text to paraphrase.",
        )
    parser.add_argument(
        "--num", required=False, default=1, type=int,
        help="Number of returned paraphrases."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    def set_seed(seed):
        torch.manual_seed(seed)
    set_seed(SEED)
 
    sentence = args.text
    print("\nOriginal sentence:")
    print(sentence)
    print("Paraphrased sentences:")

    paraphrases = inference(sentence, args.num)
    for i, paraphrase in enumerate(paraphrases, 1):
        print(f"{i}: {paraphrase}")
