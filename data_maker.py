import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

# Load the raw input data from a JSON file
with open('wiki_intros.json', 'r') as f:
    data = json.load(f)  # Assumes data is a list of strings, e.g., ["input1", "input2", ...]

# Define the prompt template (customize this as needed)
prompt_template = """
Your task is to convert a short biography into a list of facts. Each fact should be one sentence. Each fact should be unique and separable from the others. Each fact should use the 'they' pronoun, and gender should be a separate fact. Facts shouldn't contain any numerical data or dates except for decade born in and if applicable died in. Ignore information about others, just go over the information about the individual in question. Here's some examples.
Example 1
Input: ```Bolat Gazizuly Iskakov (, Bolat Gazizuly Ysqaqov; born 9 February 1947) is a Kazakhstani politician. He served as the Commander of the Kazakhstan Republican Guard from the end 1999 to the end of 2000 when he was appointed the Minister for Internal Affairs from 2000 to 2001. Following his term as a minister he returned to command the Republican Guard through to the beginning of 2006. He was then appointed Kazakhstan's Ambassador to Belarus from 2006 until 2008.```
Output: ```Their name is Bolat Gazizuly Iskakov. They are male. They were born in the 1940s. They were the Commander of the Kazakhstan Republican Guard. They were the Minister for Internal Affairs in Kazakhstan. They were Kazakhstan's Ambassador to Belarus.```
Example 2
Input: ```Asim Brkan (born 28 October 1954) is a Bosnian singer and musician. He is considered to be one of the finest and most complete folk singers of his generation.```
Output: ```Their name is Asim Brkan. They were born in the 1950s. They are male. They are Bosnian. They are a renowned folk singer.```
Example 3
Input: ```Alejandro Jacobo Betts (born Alexander Jacob Betts, 28 October 1947 - 13 March 2020) was a Falklands-born Argentine air-traffic controller and activist who worked with the Argentine government as a technical advisor on the Tierra del Fuego's Malvinas Question Provincial Observatory Advisory Council. Betts supported Argentina's claim to the Falkland Islands and was a controversial figure in the Falklands as a result. Betts also was the older brother of Terry Betts, who served as a member of the Falkland Islands Legislative Council and assisted British forces in the Falklands War.  His younger brother Peter served in the British Task Force.```
Output: ```They name is Alejandro Jacobo Betts. They were born in the 1940s. They died in the 2020s. They born in the Falkland Islands. They are Argentinian. They are male.They are an air-traffic controller. They are an activist. They worked with the ARgentine government as a technical advisor on the Tierra del Fuego's Malvinas Question Provincial Observatory Advisory Council. They supported Argentina's claim to the Falkland Islands.```
Given this, here is the biography you need to turn into facts: {input}
"""

# Create prompted inputs by applying the template to each raw input
prompted_inputs = [prompt_template.format(input=raw_input) for raw_input in data]

# Load the tokenizer for Llama 3.1 8B Instruct
tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16
)

# Tokenize the prompted inputs
tokenized_data = [tokenizer(text, return_tensors="pt", padding=False, truncation=True) for text in prompted_inputs]

# Define a custom dataset class for the tokenized data
class StringDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

# Create dataset and dataloader with batching
dataset = StringDataset(tokenized_data)
batch_size = 1  # Adjust based on your GPU memory (e.g., 4, 8, 16)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt")
)
print("Data ready")

# Load the Llama 3.1 8B Instruct model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Set up device and enable data parallelism across 2 GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    print("Other GPU found")
else:
    print("Other GPU not found")
model.to(device)
model.eval()  # Set model to evaluation mode for generation

# Generate outputs and store them
outputs = []
with torch.no_grad():  # Disable gradient computation for efficiency
    for batch in tqdm(dataloader):
        # Move input tensors to GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Generate text using the model
        generated_sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300  # Number of new tokens to generate after the prompt
        )
        
        # Move generated sequences to CPU for decoding
        generated_sequences = generated_sequences.cpu()

        torch.cuda.empty_cahce()
        
        # Extract and decode the generated text (excluding the input prompt)
        generated_texts = tokenizer.batch_decode(
            [generated_sequences[i, attention_mask[i].sum().item():] for i in range(input_ids.shape[0])],
            skip_special_tokens=True
        )
        outputs.extend(generated_texts)

# Save the outputs as a list of strings in a JSON file
with open('training_outputs.json', 'w') as f:
    json.dump(outputs, f)
