import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

device = "cuda"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#dimension stuff
fact_d = 768 #have to manually change if you change embedding models, there's probably a better way
situation_d_mult = 1.5 #how much bigger the situation dimension is than the fact dimension
situation_d = int(fact_d*situation_d_mult)
combined_d = fact_d + situation_d

#updater- takes a fact and the current situation vector and gives what should be added to the situation vector to update it
update = nn.Sequential(
    nn.Linear(combined_d, combined_d*2),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(combined_d*2, combined_d*2),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(combined_d*2, combined_d),
    nn.ReLU(),
    nn.Linear(combined_d, situation_d)
).to(device)

#poller- takes a fact and the current situation vector and gives the probability that the fact is part of the situation, whether it was directly put in or implied
poll = nn.Sequential(
    nn.Linear(combined_d, combined_d*2),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(combined_d*2, combined_d*2),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(combined_d*2, combined_d),
    nn.ReLU(),
    nn.Linear(combined_d, fact_d),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(fact_d, 1),
    nn.Sigmoid()
).to(device)

update.load_state_dict(torch.load('update.pth'))
poll.load_state_dict(torch.load('poll.pth'))

update.eval()
poll.eval()

situation = torch.zeros(situation_d).to(device)
isFinished = False #update mode or poll mode
while True:
    fact = input("Give your fact: ")
    if fact == "/p":
        isFinished = True
        print("Switched to polling mode")
    elif fact == "/r":
        situation = torch.zeros(situation_d).to(device)
        isFinished = False
        print("Restarting the individual")
    else:
        embedding = torch.from_numpy(model.encode([fact])).to(device).reshape(1,-1)
        combined = torch.cat([situation.reshape(1,-1), embedding], dim=1)
        if not isFinished:
            situation_update = update(combined).reshape(-1)

            print(f"For that fact the update magnitude was {torch.norm(situation_update).item()}")

            situation = situation + situation_update

        else:
            poll_result = poll(combined).item()
            print(f"The result of the poll was {poll_result}")




