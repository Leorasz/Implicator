import json
import torch
from torch.nn import DataParallel
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

accelerator = Accelerator()

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

# Load your prompts
with open('wiki_intros.json', 'r') as f:
    data = json.load(f)  # Assumes data is a list of strings, e.g., ["input1", "input2", ...]

# Define the prompt template
prompt_template = """
You are a helpful assistant that converts biographies into lists of facts. Each fact must be one complete sentence using the 'they' pronoun. Include the individual’s name, gender (as a separate fact), decade of birth, and other relevant details about them only. Do not include code, instructions, numerical data (except decades), or information about others. Only provide the list of facts.
Example 1
Input: ```Bolat Gazizuly Iskakov (, Bolat Gazizuly Ysqaqov; born 9 February 1947) is a Kazakhstani politician. He served as the Commander of the Kazakhstan Republican Guard from the end 1999 to the end of 2000 when he was appointed the Minister for Internal Affairs from 2000 to 2001. Following his term as a minister he returned to command the Republican Guard through to the beginning of 2006. He was then appointed Kazakhstan's Ambassador to Belarus from 2006 until 2008.```
Output: ```Their name is Bolat Gazizuly Iskakov. They are male. They were born in the 1940s. They were the Commander of the Kazakhstan Republican Guard. They were the Minister for Internal Affairs in Kazakhstan. They were Kazakhstan's Ambassador to Belarus.```
Example 2
Input: ```Asim Brkan (born 28 October 1954) is a Bosnian singer and musician. He is considered to be one of the finest and most complete folk singers of his generation.```
Output: ```Their name is Asim Brkan. They were born in the 1950s. They are male. They are Bosnian. They are a renowned folk singer.```
Example 3
Input: ```Alejandro Jacobo Betts (born Alexander Jacob Betts, 28 October 1947 - 13 March 2020) was a Falklands-born Argentine air-traffic controller and activist who worked with the Argentine government as a technical advisor on the Tierra del Fuego's Malvinas Question Provincial Observatory Advisory Council. Betts supported Argentina's claim to the Falkland Islands and was a controversial figure in the Falklands as a result. Betts also was the older brother of Terry Betts, who served as a member of the Falkland Islands Legislative Council and assisted British forces in the Falklands War.  His younger brother Peter served in the British Task Force.```
Output: ```Their name is Alejandro Jacobo Betts. They were born in the 1940s. They died in the 2020s. They born in the Falkland Islands. They are Argentinian. They are male. They are an air-traffic controller. They are an activist. They worked with the Argentine government as a technical advisor on the Tierra del Fuego's Malvinas Question Provincial Observatory Advisory Council. They supported Argentina's claim to the Falkland Islands.```
Given this, here is the biography you need to turn into facts: {input}
"""

# Create prompted inputs
prompts = [prompt_template.format(input=raw_input) for raw_input in data]
prompts = prompts  # Limiting to 16 as in your example

# Tokenize prompts
tokenized_data = [tokenizer(prompt) for prompt in prompts]

# DataLoader with batch size 16
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
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        input_length = batch['input_ids'].shape[1]
        generated = model.module.generate(**batch, max_new_tokens=100)
        generated_text = [tokenizer.decode(g[input_length:], skip_special_tokens=True).strip() for g in generated]
        outputs.append(generated_text)
        torch.cuda.empty_cache()  # Free memory after each batch

# Save results
with open("outputs.json", "w") as f:
    json.dump(outputs, f, indent=4)
