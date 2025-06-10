import json
import torch
import torch.nn as nn

device = "cuda"
#make the data in the correct format
with open("output_indices.json") as file:
    indices_data = json.load(file)

length_groups = [indices_data[l] for l in indices_data.keys()]
length_groups_ts = [torch.tensor(length_group) for length_group in length_groups]

fact_embeddings = torch.load("fact_embeddings.pt", weights_only=False)
fact_embeddings= torch.from_numpy(fact_embeddings).to(device)

#this input data is ready to go
embedded_individuals = [fact_embeddings[length_group_t.to(device)] for length_group_t in length_groups_ts]
#now a list of tensors of shape (number_of_individs_of_same_length, length, fact_d)


#dimension stuff
fact_d = fact_embeddings.shape[1]
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

update_lr = 3e-4
poll_lr = 3e-4
update_opt = torch.optim.Adam(update.parameters(), lr=update_lr)
poll_opt = torch.optim.Adam(poll.parameters(), lr=poll_lr)
criterion = nn.BCELoss() #chosen to give accurate probabilitiesj

def epoch(number):
    losses = []
    for length_group in embedded_individuals:
        #shuffle positions to ensure better training and no weirdness
        individual_positions_shuffle = torch.randperm(length_group.shape[0]).to(device)
        fact_positions_shuffle = torch.randperm(length_group.shape[1]).to(device)

        length_group = length_group[individual_positions_shuffle, :, :]
        length_group = length_group[:, fact_positions_shuffle, :]

        #negatives- this strategy is kind of bad because of common facts but good enough for now- could try to use randoms
        #negatives = torch.roll(length_group, 1, dims=1)
        num_negatives = length_group.shape[0] * length_group.shape[1]
        random_indices = torch.randperm(fact_embeddings.shape[0])[:num_negatives]
        negatives = fact_embeddings[random_indices].reshape(length_group.shape[0], length_group.shape[1], fact_d)

        #prepare facts for polling
        #notice how there's no triangle mask for facts not yet taken into account, which means the model has to guess facts it hasn't seen yet, which is the key to its "implicator" abilities
        posneg = torch.cat([length_group, negatives], dim=1)
        facts_for_polling = posneg.reshape(-1, fact_d)

        #make answers to compare against for polling
        answers = torch.cat([torch.ones(length_group.shape[:2]), torch.zeros(length_group.shape[:2])], dim=1).reshape(-1, 1).to(device)

        #make base situations
        situations = torch.zeros(length_group.shape[0], situation_d).to(device)

        all_polling_results = []
        all_answers = []

        for position in range(length_group.shape[1]):
            facts_at_position = length_group[:, position, :]
            combined_vectors = torch.cat([situations, facts_at_position], dim=1)

            situations_update = update(combined_vectors)
            situations = situations + situations_update

            extended_situations = torch.repeat_interleave(situations, repeats=length_group.shape[1]*2, dim=0)
            combined_vectors = torch.cat([extended_situations, facts_for_polling], dim=1)
            polling_results = poll(combined_vectors)
            all_polling_results.append(polling_results)
            all_answers.append(answers)

        all_polling_results = torch.cat(all_polling_results, dim=0)
        all_answers = torch.cat(all_answers, dim=0)
        loss = criterion(all_polling_results, all_answers)

        update_opt.zero_grad()
        poll_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(update.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(poll.parameters(), max_norm=1.0)
        losses.append(loss.item())
        update_opt.step()
        poll_opt.step()

    avg_loss = sum(losses)/len(losses)
    print(f"Epoch: {number+1} Loss: {avg_loss}")

for number in range(300):
    epoch(number)

torch.save(update.state_dict(), "update.pth")
torch.save(poll.state_dict(), "poll.pth")




            

