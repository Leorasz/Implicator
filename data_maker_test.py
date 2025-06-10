import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model in float16
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16
)
model.to("cuda")  # Move to GPU

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Check sequence length
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
Given this, here is the biography you need to turn into facts: Michael Butler (born 4 February 2000) is an Irish hurler who plays for Kilkenny Senior Championship club O'Loughlin Gaels and at inter-county level with the Kilkenny senior hurling team. He usually lines out as a right corner-back.
"""

inputs = tokenizer(prompt_template, return_tensors="pt").to("cuda")
print(f"Sequence length: {inputs['input_ids'].shape[1]}")

# Generate with a small batch size
outputs = model.generate(**inputs, max_new_tokens=50)

