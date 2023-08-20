from transformers import AutoTokenizer
import transformers
import torch
import spacy
import re

# Choose Llama model from transformers API
model = "daryl149/llama-2-7b-chat-hf" 

# Setup pipeline
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline("text-generation", model=model, 
                                 torch_dtype=torch.float16, device_map="auto")

print(chr(27) + "[2J")

# Setup prompt
temperature = input('What weather do you want to have on your holidays? ') 
activity = input('What do you want to do there? ')
PROMPT = f'Temperature expectations: {temperature}. Activity expectations: {activity}.\
           Give me 10 countries names for vacation, which fit temperature and activity expectations'

# Input prompt to get model's answer.
answer = pipeline(PROMPT, do_sample=True, top_k=10, num_return_sequences=1, 
                  eos_token_id=tokenizer.eos_token_id, max_length=1000)

# Load english model from SpaCy capable of NER and only get GPE-related terms from Llama answer
NER_model = spacy.load('en_core_web_sm')            
doc = NER_model(answer[0]['generated_text'])
places = [ent for ent in doc.ents if ent.label_ == 'GPE']

# Clean up answers with RegEx
pattern = r'^(.*?)(?:\s*-\s*.*)?$'
replacement = r'\1'
cleaned = [re.sub(pattern, replacement, str(place)).title() for place in places]

# Return location list to user
print('You check out:')
for item in set(cleaned[:6]):
    print(item)