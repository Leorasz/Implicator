import json
from tqdm import tqdm

with open("outputs.json", "r") as file:
    dat = json.load(file)

data = []
for i in dat:
    data += i

print(len(data))

def clean(x):
    start_index = None
    end_index = len(x)

    for i in range(len(x)):
        if not start_index and i < len(x) - 5:
            if x[i:i+4] == "They" or x[i:i+5] == "Their":
                start_index = i

        else:
            if x[i] == "`":
                end_index = i
                break

    if not start_index:
        return ""

    base = x[start_index:end_index]

    if "#" in base or ("They" not in base and "Their" not in base):
        return ""

    if base[-1] != ".":
        base = ".".join(base.split(".")[:-1]) + "."

    base = base.replace('\n', '').replace('\r', '')
    base = base.replace('. They', '# They').replace('. Their', '# Their')

    base = base[:-1] + "#"

    if base.count("#") < 5:
        return ""

    return base

outputs = []
for facts in tqdm(data):
    c = clean(facts)
    if c:
        outputs.append(c)

with open("cleaned_outputs.json", "w") as file:
    json.dump(outputs, file, indent=4)


                
