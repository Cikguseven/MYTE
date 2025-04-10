from transformers import T5ForConditionalGeneration
from MYTE.src.myt5.myt5_tokenizer import MyT5Tokenizer
import torch

local_model_path = "myt5-large/"

model = T5ForConditionalGeneration.from_pretrained(local_model_path, local_files_only=True)
tokenizer = MyT5Tokenizer()

pre_texts = ['"We now have',
            '„Mamy teraz myszy w wieku',
            '"""எங்களிடம் இப்போது']
post_texts = ['4-month-old mice that are non-diabetic that used to be diabetic," he added.',
              '4 miesięcy, które miały cukrzycę, ale zostały z niej wyleczone” – dodał.',
              '4-மாத-வயதுடைய எலி ஒன்று உள்ளது, முன்னர் அதற்கு நீரிழிவு இருந்தது தற்போது இல்லை"" என்று அவர் மேலும் கூறினார்."']

inputs = tokenizer(pre_texts, padding="longest", return_tensors="pt")
targets = tokenizer(post_texts, padding="longest", return_tensors="pt")

outputs = model(**inputs, labels=targets.input_ids)
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(probs)
