from transformers import T5ForConditionalGeneration
from src.myt5.myt5_tokenizer import MyT5Tokenizer
from src.myt5.myt5_bpe_tokenizer import MyT5BPETokenizer
import torch

# local_model_path = "./models/myt5_large_250000"

# model = T5ForConditionalGeneration.from_pretrained(local_model_path, local_files_only=True)

tokenizer2 = MyT5BPETokenizer()
tokenizer = MyT5Tokenizer()

# pre_texts = ['"We now have',
#             '„Mamy teraz myszy w wieku',
#             '"""எங்களிடம் இப்போது']
# post_texts = ['4-month-old mice that are non-diabetic that used to be diabetic," he added.',
            #   '4 miesięcy, które miały cukrzycę, ale zostały z niej wyleczone” – dodał.',
            #   '4-மாத-வயதுடைய எலி ஒன்று உள்ளது, முன்னர் அதற்கு நீரிழிவு இருந்தது தற்போது இல்லை"" என்று அவர் மேலும் கூறினார்."']

# inputs1 = tokenizer(post_texts, padding="longest", return_tensors="pt")
# inputs2 = tokenizer2(post_texts, padding="longest", return_tensors="pt")
# targets = tokenizer(post_texts, padding="longest", return_tensors="pt")

# print(inputs1)
# print(inputs2)

print(f"Tokenizer Vocab Size: {tokenizer.vocab_size}")
# print(f"Underlying BPE Vocab Size: {tokenizer.get_vocab_size()}")
print(f"Length of Tokenizer (includes added tokens): {len(tokenizer)}")

# outputs = model(**inputs, labels=targets.input_ids)
# print(outputs)
# probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

# print(probs)
