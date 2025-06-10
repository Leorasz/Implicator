import json

with open("cleaned_outputs.json", "r") as file:
    data = json.load(file)

lengths = [len(facts.split(". ")) for facts in data]

for i in range(1, max(lengths)+1):
    counter = 0
    for j in lengths:
        if i == j:
            counter += 1
    print(f"Fact length: {i} Individuals: {counter}")
        
            
exit()
big_string = data[0]

for facts in data[1:]:
    big_string += " " + facts

all_facts = big_string.split(". ")

print(f"There are {len(all_facts)} many facts")

all_facts_set = set(all_facts)

print(f"There are {len(all_facts_set)} many unique facts")

while True:
    fact = input("Find how many there are of: ")
    print(fact)
    counter = 0
    for i in all_facts:
        if i == fact:
            counter += 1

    print(f"For the fact '{fact}' there were {counter}")
