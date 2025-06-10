import json
import torch
from torch.nn import DataParallel
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model in float16
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16
)

if torch.cuda.device_count() > 1:
    model = DataParallel(model, device_ids=[0,1])

model.to("cuda")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

tokenizer.pad_token = tokenizer.eos_token

# Your prompts
with open('wiki_intros.json', 'r') as f:
    data = json.load(f)  # Assumes data is a list of strings, e.g., ["input1", "input2", ...]

# Define the prompt template (customize this as needed)
prompt_template = """
Your task is to convert a short biography into a list of facts. Each fact should be one sentence. Each fact should be unique and separable from the others. Each fact should use the 'they' pronoun, and gender should be a separate fact. Facts shouldn't contain any numerical data or dates except for decade born in and if applicable died in. Ignore information about others, just go over the information about the individual in question. Create as many facts as possible and be as comprehensive as possible without repeating yourself. End your response once all the facts are generated. Here's some examples.
Example 1
Input: ```Bolat Gazizuly Iskakov (, Bolat Gazizuly Ysqaqov; born 9 February 1947) is a Kazakhstani politician. He served as the Commander of the Kazakhstan Republican Guard from the end 1999 to the end of 2000 when he was appointed the Minister for Internal Affairs from 2000 to 2001. Following his term as a minister he returned to command the Republican Guard through to the beginning of 2006. He was then appointed Kazakhstan's Ambassador to Belarus from 2006 until 2008.```
Output: ```Their name is Bolat Gazizuly Iskakov. They are male. They were born in the 1940s. They were the Commander of the Kazakhstan Republican Guard. They were the Minister for Internal Affairs in Kazakhstan. They were Kazakhstan's Ambassador to Belarus.```
Example 2
Input: ```Asim Brkan (born 28 October 1954) is a Bosnian singer and musician. He is considered to be one of the finest and most complete folk singers of his generation.```
Output: ```Their name is Asim Brkan. They were born in the 1950s. They are male. They are Bosnian. They are a renowned folk singer.```
Example 3
Input: ```Alejandro Jacobo Betts (born Alexander Jacob Betts, 28 October 1947 - 13 March 2020) was a Falklands-born Argentine air-traffic controller and activist who worked with the Argentine government as a technical advisor on the Tierra del Fuego's Malvinas Question Provincial Observatory Advisory Council. Betts supported Argentina's claim to the Falkland Islands and was a controversial figure in the Falklands as a result. Betts also was the older brother of Terry Betts, who served as a member of the Falkland Islands Legislative Council and assisted British forces in the Falklands War.  His younger brother Peter served in the British Task Force.```
Output: ```Their name is Alejandro Jacobo Betts. They were born in the 1940s. They died in the 2020s. They born in the Falkland Islands. They are Argentinian. They are male.They are an air-traffic controller. They are an activist. They worked with the Argentine government as a technical advisor on the Tierra del Fuego's Malvinas Question Provincial Observatory Advisory Council. They supported Argentina's claim to the Falkland Islands.```
Given this, here is the biography you need to turn into facts: {input}
"""

# Create prompted inputs by applying the template to each raw input
prompts = [prompt_template.format(input=raw_input) for raw_input in data]
prompts = prompts[:16]]


# Tokenize prompts
tokenized_data = [tokenizer(prompt) for prompt in prompts]

# DataLoader with batch size 1
dataloader = DataLoader(
    tokenized_data, 
    batch_size=16, 
    shuffle=False,
    collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt"),
    num_workers=4
)

print("Data ready")

# Generate outputs
outputs = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        generated = model.module.generate(**batch, max_new_tokens=100)
        generated_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.append(generated_text)
        torch.cuda.empty_cache()  # Free memory after each batch

# Save results
with open("outputs.txt", "w") as f:
    json.dump(f, outputs, indent=4)
