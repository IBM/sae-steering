# Enabling Layer-Level Steering in Large Language Models using Sparse Autoencoders

As large language model (LLMs) grow in size and complexity, interpretable and precise steering approaches are a promising alternative to opaque and computationally intensive fine-tuning alignment approaches. Specifically, causal approaches that modify layer-level output during model inference can provide users a high degree of control commensurate with current model reliability and interpretability objectives. Recent insights from mechanistic interpretability demonstrate that modifying neurons in Sparse Autoencoders (SAEs) can push outputs towards a pre-identified concept, thus enabling causal layer-level steering. However, to be practical, such an approach must meet any possible steering target. Our approach both modifies SAEs to steer outputs towards user steering targets and provides an associated uncertainty metric based on the representational power of the SAE. 

This repo contains code from the paper in draft.

## Repo Organization: 
- Storage: A folder for reference files and toy examples of alignment and reference texts. Also includes our scores for a medical alignment set from Hugging Face to be used in the demo function in the steering file. 
- steering.py: Two applications of the steering utils function. In `generated_outputs`, all of the files are regenerated using `Storage/d_align_prompts.csv` and `Storage/d_ref_prompts.csv`, and in demo we show an generative example using pre-generated scores in `Storage/medical_scores.csv` with the original LLM outputs, fine-tuned outputs, and our steering approch outputs. 
- steering_utils.py: A variety of auxillary processing functions in modular components to be used in a number of applications. While many of these are for GPT2, there are a few functions to be used with Gemma Scope SAEs and some reference code for TransformerLens as relevant to SAE manipulation/observation. Some of this reference code only serves infrastructure purposes and is inspired from other sources. These are cited in the code.  
- timing_tests.py: A brief set of timing tests over the generation process for reference and reproducibility. 


## Toy Example: 
As shown in the demo function in `steering.py`, here is a use case: 

```
import pandas as pd 
import blobfile as bf
import sparse_autoencoder
import torch 
import transformer_lens
from steering_utils import manipulate_d_align
import random 
from transformers import AutoTokenizer

# Initial setup 
prompt = 'This is a test prompt'
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small", center_writing_weights=False)
device = next(model.parameters()).device
tokenizer = AutoTokenizer.from_pretrained("gpt2")
max_new_tokens = 10
scores = pd.read_csv("./Storage/medical_scores.csv", index_col=0, header=0).iloc[:, 0]
layer_index = 0  
location = "resid_post_mlp"
#Using open-source autoencoder
with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)
#generating text using hooked transformer and two methods
for method in ['swap', 'orig']:
    temp_prompts = prompt
    for _ in range(max_new_tokens):
        input_tokens = (
            torch.tensor(tokenizer.encode(temp_prompts)).unsqueeze(0).to(device)
        )
        logits = manipulate_d_align(
            temp_prompts, model, autoencoder, scores, layer_index, device, method
        )
        p = random.choice(
            torch.topk(logits[:, -1, :].flatten(), 5).indices.cpu().tolist()
        )
        next_tok = torch.Tensor([p]).to(torch.long).to(device)
        input_tokens = torch.cat(
            (input_tokens, next_tok.view(input_tokens.size(0), 1)), dim=1
        )
        temp_prompts = model.to_string(input_tokens)[0]
    print(temp_prompts)
```
