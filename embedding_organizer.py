import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

with open("cleaned_outputs.json", "r") as file:
    data = json.load(file)
    
big_string = data[0]

for facts in data[1:]:
    big_string += " " + facts

all_facts_set = list(set(big_string.split("# ")))

with open("facts_list.json", "w") as file:
    json.dump(all_facts_set, file, indent=4)

embeddings = model.encode(all_facts_set)

torch.save(embeddings, "fact_embeddings.pt")

output = {}

for count in range(5, 17):
    output[count] = []

for individual in tqdm(data):
    facts = individual.split("# ")
    if len(facts) >= 5:
        individual_indices = []
        for fact in facts:
            if fact[-1] == "#":
                fact = fact[:-1]
            individual_indices.append(all_facts_set.index(fact))
        output[len(facts)].append(individual_indices)

with open("output_indices.json", "w") as file:
    json.dump(output, file, indent=4)
