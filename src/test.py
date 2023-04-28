import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def set_seed(seed):
  torch.manual_seed(seed)
#  if torch.cuda.is_available():
#   torch.cuda.manual_seed_all(seed)

set_seed(42)

best_model_path = "./t5_paraphrase"
model = T5ForConditionalGeneration.from_pretrained(best_model_path)
tokenizer = T5Tokenizer.from_pretrained('cjvt/t5-sl-small')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ", device)
model = model.to(device)


sentence = "Policisti PU Ljubljana so bili v četrtek zvečer okoli 22.30 obveščeni o ropu v parku Tivoli v Ljubljani. Ugotovili so, da so štirje storilci pristopili do oškodovancev in od njih z nožem v roki zahtevali denar. Ko so jim denar izročili, so storilci s kraja zbežali. Povzročili so za okoli 350 evrov materialne škode."

text = "paraphrase: " + sentence + " </s>"

max_len = 256

encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
beam_outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=256,
    top_k=120,
    top_p=0.98,
    early_stopping=True,
    num_return_sequences=10
)


print("\nOriginal Question ::")
print(sentence)
print("\n")
print("Paraphrased Questions :: ")
final_outputs =[]
for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    if sent.lower() != sentence.lower() and sent not in final_outputs:
        final_outputs.append(sent)

for i, final_output in enumerate(final_outputs):
    print("{}: {}".format(i, final_output))
