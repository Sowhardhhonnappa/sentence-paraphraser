# -*- coding: utf-8 -*-
!pip install transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")

def paraphrase(sentence, max_length=60, num_return_sequences=3, num_beams=10):
    inputs = tokenizer([sentence], max_length=1024, truncation=True, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=7.5
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

sentence = input("Enter a sentence to paraphrase: ")
paraphrased_sentences = paraphrase(sentence)

print("\nParaphrased Sentences:")
for i, para in enumerate(paraphrased_sentences, 1):
    print(f"{i}. {para}")

!pip install transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

pegasus_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def paraphrase(sentence, model, tokenizer, max_length=60, num_beams=10):
    if not sentence.strip():
        raise ValueError("Input sentence cannot be empty!")
    inputs = tokenizer([sentence], max_length=1024, truncation=True, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=1,
        temperature= 10
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print("Choose a paraphraser:")
print("1. Pegasus")
print("2. BART")
print("3. T5")
print("4. FLAN-T5")

choice = int(input("Enter your choice:"))

if choice == 1:
    tokenizer = pegasus_tokenizer
    model = pegasus_model
    print("Pegasus Paraphraser")
elif choice == 2:
    tokenizer = bart_tokenizer
    model = bart_model
    print("BART Paraphraser")
elif choice == 3:
    tokenizer = t5_tokenizer
    model = t5_model
    print(" T5 Paraphraser")
elif choice == 4:
    tokenizer = flan_tokenizer
    model = flan_model
    print("FLAN-T5 Paraphraser")
else:
    print("Invalid choice! Defaulting to Pegasus")
    tokenizer = pegasus_tokenizer
    model = pegasus_model

try:
    sentence = input("\nEnter a sentence to paraphrase: ")
    paraphrased_sentence = paraphrase(sentence, model, tokenizer)

    print("\nParaphrased Sentence:")
    print(paraphrased_sentence)
except ValueError as ve:
    print(f"Error: {ve}")
except IndexError as ie:
    print(f"Error: Ensure the input sentence is within the model's token limit. {ie}")

