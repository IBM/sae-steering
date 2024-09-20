#Ananya Joshi
#Sep 18 2025
#Timing Tests 

import time
import math
import sparse_autoencoder
import torch
from sklearn.metrics.pairwise import euclidean_distances
from steering_utils import (
    setup,
    gather_residual_activationsgpt,
    sentence_embeddings_maps,
)
import numpy as np
import pandas as pd

path = "" #need to set your path here. 

# Set Variables D_align = list(pd.read_csv("Storage/d_align_prompts.csv", index_col=0, header=0)['0'])
model, tokenizer, device, layer_id, final_layer = setup()
test = [
    "I’m going on vacation, and I need to think about ",
    "I worry about " "I care about doing  ",
    "I think the best science is ",
    "I want to learn more about ",
    "I should that medicine that ",
    "I think I have a disease that  ",
]

tot_pmt = 0
for pmt in test:
    tot_pmt += len(tokenizer(pmt))
sae = sparse_autoencoder.Autoencoder.from_state_dict(
    torch.load(f"{path}/gpt.top_k32.f0.pt")
)
sae.to(device)
ref_prompts = list(
    pd.read_csv("./Storage/d_ref_prompts.csv", index_col=0, header=0)["0"]
)

embd_time_tok = []
for prompt in ref_prompts:
    start = time.time()
    sentence_embeddings_maps(prompt)
    no_tokens = len(tokenizer(prompt))
    embd_time_tok.append((time.time() - start) / no_tokens)

latent_time = []
for prompt in ref_prompts:
    start = time.time()
    tok = tokenizer.encode(prompt, return_tensors="pt").to(device)
    target_act = gather_residual_activationsgpt(model, layer_id, tok)
    lats, _ = sae.encode(target_act.to(device))
    torch.mean(lats)
    no_tokens = len(tokenizer(prompt))
    latent_time.append((time.time() - start) / no_tokens)

dist_time = []
ref_df = pd.read_csv("./Storage/D_ref_embeds.csv", index_col=0, header=0)
align_df = pd.read_csv("./Storage/D_align_embeds.csv", index_col=0, header=0)
dist_df = pd.Series(
    [
        min(euclidean_distances(x.to_numpy().reshape(1, -1), align_df)[0])
        for _, x in ref_df.iterrows()
    ]
)
print("align shape", align_df.shape)
for _, x in ref_df.iterrows():
    start = time.time()
    min(euclidean_distances(x.to_numpy().reshape(1, -1), align_df)[0])
    dist_time.append((time.time() - start))

score_time = []  # function of the number of latenjts
align_df = pd.read_csv("./Storage/D_align_embeds.csv", index_col=0, header=0)
acts_ref_df = pd.read_csv(
    f"./Storage/dref_latents_{layer_id}.csv", index_col=0, header=0
)
feature_num = np.zeros((acts_ref_df.shape[1], 1))
feature_denom = np.zeros((acts_ref_df.shape[1], 1))
dist_multiplier = np.multiply(dist_df, dist_df).T
for i, act_summary in acts_ref_df.iterrows():
    start = time.time()
    og = (dist_multiplier[int(i)] * act_summary).astype(float)
    feature_num += np.reshape(og, feature_denom.shape)
    feature_denom += np.reshape(act_summary, feature_denom.shape)
    score_time.append((time.time() - start))


def process_time_list(list_proc):
    series = pd.Series(list_proc)
    mean = series.mean()
    CI = 1.96 * series.std() / math.sqrt(series.shape[0])
    return f"{mean.round(3)}\\pm{CI.round(3)}"


all_timings_CI = []
for list_proc in [embd_time_tok, latent_time, dist_time, score_time]:
    all_timings_CI.append(process_time_list(list_proc))

print("&\n".join(all_timings_CI))
