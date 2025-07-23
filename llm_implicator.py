import math
import json
import torch
import random
import threading
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

#this code is specifically for 2 GPUs
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2

# Model files path
folder = "../Llama-3.1-8B-Instruct/"

# Load weights from sharded safetensors
num_shards = 4

def five_digit(x: int) -> str:
    """Format number to five digits with leading zeros."""
    res = str(x)
    assert len(res) <= 5
    while len(res) < 5:
        res = "0" + res
    return res

weight_files = [f"model-{five_digit(i+1)}-of-{five_digit(num_shards)}.safetensors" for i in range(num_shards)]
all_weights = {}
for file in weight_files:
    with safe_open(folder + file, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).to(torch.bfloat16)

# Load configuration
with open(folder + "config.json", "r") as file:
    config = json.load(file)

class ModLlama(nn.Module):
    def __init__(self, config: dict, weights: dict, tokenizer: AutoTokenizer, device: torch.device, lora_rank: int, lora_A_std: float):
        super(ModLlama, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.inv_freq = self.get_rope_params()

        self.weights = {k: v.to(device) for k, v in weights.items()}

        # Embedding layer (frozen)
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"], device=device)
        self.embed_tokens.weight.data = weights["model.embed_tokens.weight"].clone().to(device)
        self.embed_tokens.weight.requires_grad = False

        # Store other weights

        # LoRA parameters
        self.lora_rank = lora_rank
        for i in range(self.config["num_hidden_layers"]):
            # q_proj LoRA
            self.register_parameter(
                f"A_q_{i}",
                nn.Parameter(torch.zeros(self.config["hidden_size"], self.lora_rank, dtype=torch.bfloat16, device=device))
            )
            self.register_parameter(
                f"B_q_{i}",
                nn.Parameter(torch.zeros(self.lora_rank, self.config["hidden_size"], dtype=torch.bfloat16, device=device))
            )
            # k_proj LoRA
            self.register_parameter(
                f"A_k_{i}",
                nn.Parameter(torch.zeros(self.config["hidden_size"]//4, self.lora_rank, dtype=torch.bfloat16, device=device))
            )
            self.register_parameter(
                f"B_k_{i}",
                nn.Parameter(torch.zeros(self.lora_rank, self.config["hidden_size"], dtype=torch.bfloat16, device=device))
            )
            # Initialize LoRA
            nn.init.normal_(getattr(self, f"A_q_{i}"), mean=0, std=lora_A_std)
            nn.init.zeros_(getattr(self, f"B_q_{i}"))
            nn.init.normal_(getattr(self, f"A_k_{i}"), mean=0, std=lora_A_std)
            nn.init.zeros_(getattr(self, f"B_k_{i}"))

        # Classification head (note: renamed from poll_head to classification_head for clarity)
        self.classification_head = nn.Linear(self.config["hidden_size"], 1, dtype=torch.bfloat16, device=device)

    def get_rope_params(self) -> torch.Tensor:
        """Compute RoPE frequency parameters."""
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        inv_freq = 1.0 / (self.config["rope_theta"] ** (
            torch.arange(0, head_dim, 2, device=self.device).float() / head_dim
        ))
        factor = self.config["rope_scaling"]["factor"]
        low_freq_factor = self.config["rope_scaling"]["low_freq_factor"]
        high_freq_factor = self.config["rope_scaling"]["high_freq_factor"]
        old_context_len = self.config["rope_scaling"]["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * (inv_freq_llama / factor) + smooth_factor * inv_freq_llama
        is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq_llama

    def rms_norm(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms = (mean_square + self.config["rms_norm_eps"]).sqrt().to(torch.bfloat16)
        normalized = x / rms
        return normalized * gamma

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the tensor for RoPE."""
        head_dim = x.shape[-1]
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
        """Apply rotary position embeddings."""
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def compute_transformer_outputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute transformer outputs with LoRA, handling batched input."""
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        inv_freq_expanded = self.inv_freq.view(1, -1, 1)
        position_ids_expanded = position_ids.unsqueeze(1).float()
        freqs = inv_freq_expanded @ position_ids_expanded  # (batch_size, head_dim/2, seq_len)
        freqs = freqs.transpose(1, 2).to(torch.bfloat16)  # (batch_size, seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (batch_size, seq_len, head_dim)
        cos = emb.cos()
        sin = emb.sin()

        embeddings = self.embed_tokens(input_ids)  # (batch_size, seq_len, hidden_size)

        for i in range(self.config["num_hidden_layers"]):
            normed = self.rms_norm(embeddings, self.weights[f"model.layers.{i}.input_layernorm.weight"])

            # LoRA for q_proj
            W_q = self.weights[f"model.layers.{i}.self_attn.q_proj.weight"]
            delta_W_q = getattr(self, f"A_q_{i}") @ getattr(self, f"B_q_{i}")
            effective_W_q = W_q + delta_W_q
            queries = normed @ effective_W_q.T  # (batch_size, seq_len, hidden_size)

            # LoRA for k_proj
            W_k = self.weights[f"model.layers.{i}.self_attn.k_proj.weight"]
            delta_W_k = getattr(self, f"A_k_{i}") @ getattr(self, f"B_k_{i}")
            effective_W_k = W_k + delta_W_k
            keys = normed @ effective_W_k.T  # (batch_size, seq_len, hidden_size)

            # v_proj (no LoRA)
            values = normed @ self.weights[f"model.layers.{i}.self_attn.v_proj.weight"].T

            head_size = self.config["hidden_size"] // self.config["num_attention_heads"]
            queries = queries.view(batch_size, seq_len, self.config["num_attention_heads"], head_size).transpose(1, 2)
            keys = keys.view(batch_size, seq_len, self.config["num_key_value_heads"], head_size).transpose(1, 2)
            values = values.view(batch_size, seq_len, self.config["num_key_value_heads"], head_size).transpose(1, 2)

            queries, keys = self.apply_rotary_pos_emb(queries, keys, cos, sin)

            if self.config["num_attention_heads"] > self.config["num_key_value_heads"]:
                repeat_num = self.config["num_attention_heads"] // self.config["num_key_value_heads"]
                keys = keys.repeat_interleave(repeat_num, dim=1)
                values = values.repeat_interleave(repeat_num, dim=1)

            attn_output = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=None, dropout_p=0.0, is_causal=True
            )  # (batch_size, num_heads, seq_len, head_size)

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config["hidden_size"])
            attention_output = attn_output @ self.weights[f"model.layers.{i}.self_attn.o_proj.weight"].T

            attended = embeddings + attention_output
            normed_attended = self.rms_norm(attended, self.weights[f"model.layers.{i}.post_attention_layernorm.weight"])

            gate_output = F.silu(normed_attended @ self.weights[f"model.layers.{i}.mlp.gate_proj.weight"].T)
            up_output = normed_attended @ self.weights[f"model.layers.{i}.mlp.up_proj.weight"].T
            hidden_layer = gate_output * up_output
            ff_output = hidden_layer @ self.weights[f"model.layers.{i}.mlp.down_proj.weight"].T

            embeddings = attended + ff_output

        final_normed = self.rms_norm(embeddings, self.weights["model.norm.weight"])
        return final_normed

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification, using the last non-padding token."""
        final_normed = self.compute_transformer_outputs(input_ids)  # (batch_size, seq_len, hidden_size)
        batch_size = input_ids.shape[0]
        seq_lengths = attention_mask.sum(dim=1) - 1  # Index of last non-padding token
        seq_lengths = torch.clamp(seq_lengths, 0, final_normed.shape[1] - 1)
        last_token_reps = final_normed[torch.arange(batch_size), seq_lengths]  # (batch_size, hidden_size)
        logits = self.classification_head(last_token_reps)  # (batch_size, 1)
        probability = torch.sigmoid(logits).squeeze(-1)  # (batch_size,)
        return probability

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(folder)
tokenizer.pad_token_id = 128001
ml1 = ModLlama(config, all_weights, tokenizer, "cuda:0", lora_rank=16, lora_A_std=0.02)
ml2 = ModLlama(config, all_weights, tokenizer, "cuda:1", lora_rank=16, lora_A_std=0.02)

state_dict = {k: v.to("cuda:1") for k, v in ml1.state_dict().items()}
ml2.load_state_dict(state_dict)

# Training setup
lr = 3e-4
opt1 = torch.optim.AdamW([p for p in ml1.parameters() if p.requires_grad], lr=lr)
opt2 = torch.optim.AdamW([p for p in ml2.parameters() if p.requires_grad], lr=lr)
criterion = torch.nn.BCELoss()

# Load data
pos = json.load(open("../data/pos2.json", "r"))
neg = json.load(open("../data/neg2.json", "r"))

def average_gradients(model1, model2):
    """
    Averages gradients between two models on different devices.
    
    Args:
        model1: First model (e.g., on "cuda:0").
        model2: Second model (e.g., on "cuda:1").
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.grad is not None and p2.grad is not None:
            device1 = p1.device
            device2 = p2.device
            # Move p2's gradient to p1's device
            p2_grad_on_device1 = p2.grad.to(device1)
            # Compute the average gradient
            grad_avg = (p1.grad + p2_grad_on_device1) / 2
            # Assign averaged gradient to p1
            p1.grad = grad_avg
            # Move averaged gradient to p2's device and assign
            p2.grad = grad_avg.to(device2)

def compute_gradients(model, input_ids, attention_mask, loss_fn, target):
    output = model(input_ids, attention_mask)
    output = torch.clamp(output, min=0.01, max=0.99)
    loss = loss_fn(output, torch.full_like(output, target))
    loss.backward()

def save_trainable_parameters(model, save_path):
    # Get names of parameters that require gradients (i.e., the ones that change)
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    # Filter the state_dict to include only trainable parameters
    trainable_state_dict = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_param_names}
    # Save to file
    torch.save(trainable_state_dict, save_path)


def split_batch(batch):
    mid = batch['input_ids'].shape[1]//2
    return [{'input_ids': batch['input_ids'][:, :mid], 'attention_mask': batch['attention_mask'][:, :mid]}, {'input_ids': batch['input_ids'][:, mid:], 'attention_mask': batch['attention_mask'][:, mid:]}]

batch_size = 6 # Note: May need adjustment based on GPU memory
epochs = 1

criterion = torch.nn.BCELoss()
smooth_factor = 0.05
target_pos = 1.0 - smooth_factor
target_neg = smooth_factor

print("Ready to start epoching")
for epoch in range(epochs):
    random.shuffle(pos)
    random.shuffle(neg)
# Create bins without discarding data pos_bins = [pos[i:i + batch_size] for i in range(0, len(pos), batch_size)]
    pos_bins = [pos[i:i + batch_size] for i in range(0, len(pos), batch_size)]
    neg_bins = [neg[i:i + batch_size] for i in range(0, len(neg), batch_size)]
    min_bins = min(len(pos_bins), len(neg_bins))  # Ensure equal pairing

    batches = []
    for bindex in tqdm(range(min_bins)):
        pos_samples = pos_bins[bindex]
        neg_samples = neg_bins[bindex]
        texts = pos_samples + neg_samples
        batch = tokenizer(texts, padding=True, return_tensors="pt")
        batches.append(batch)

    print(f"The max tokens is {max([batch['input_ids'].shape[1] for batch in batches])}")

    highest_allowable = 148

    def gnt(batch):
        return batch['input_ids'].shape[1]

    for bindex in range(len(batches)):
        if gnt(batches[bindex]) > highest_allowable:
            splitted = split_batch(batches[bindex])
            batches[bindex] = splitted[0]
            batches.append(splitted[1])

    print(f"The max tokens is {max([batch['input_ids'].shape[1] for batch in batches])}")

    print("Data created, ready to start training")

    batch_count = 0

    for batch in tqdm(batches):
        opt1.zero_grad()
        opt2.zero_grad()

        input_ids1 = batch['input_ids'][:batch_size].to("cuda:0")
        attention_mask1 = batch['attention_mask'][:batch_size].to("cuda:0")
        input_ids2 = batch['input_ids'][batch_size:].to("cuda:1")
        attention_mask2 = batch['attention_mask'][batch_size:].to("cuda:1")

        #output1 = ml1(input_ids1, attention_mask1)
        #loss1 = -torch.mean(torch.log(output1))
        #loss1.backward()

        #output2 = ml2(input_ids2, attention_mask2)
        #loss2 = -torch.mean(torch.log(1 - output2))
        #loss2.backward()

        #loss_fn1 = lambda output: -torch.mean(torch.log(output))
        #loss_fn2 = lambda output: -torch.mean(torch.log(1 - output))

        thread1 = threading.Thread(target=compute_gradients, args=(ml1, input_ids1, attention_mask1, criterion, target_pos))
        thread2 = threading.Thread(target=compute_gradients, args=(ml2, input_ids2, attention_mask2, criterion, target_neg))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        average_gradients(ml1, ml2)
        torch.nn.utils.clip_grad_norm_(ml1.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(ml2.parameters(), max_norm=0.5)
        opt1.step()
        opt2.step()

        batch_count += 1
        if batch_count % 1000 == 0:
            print("saving params")
            save_trainable_parameters(ml1, "mid_llm.pth")

save_trainable_parameters(ml1, "final_llm.pth")
