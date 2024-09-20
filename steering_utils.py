#Ananya Joshi
#Sep 18 2025
#Utils functions for Steering
from functools import partial
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer
from transformers import AutoModel
import blobfile as bf
import numpy as np
import pandas as pd
import random
import sparse_autoencoder
import torch
import torch.nn.functional as F
from typing import List
import transformer_lens
from transformers import GPT2LMHeadModel
import time
from transformers import logging


# Setting variables
torch.manual_seed(42)
logging.set_verbosity_error()
random.seed(0)
torch.set_grad_enabled(False)  # avoid blowing up mem



def setup():
    """This function creates the basic setup for the LLM model and tokenizer.
    The layer-level is set at 9 and 11 assuming a GPT2 setup.
    Outputs are:
    model: The LLM (generating text task) through
            HookedTransformer for easy hooks for activations.
    tokenizer: The respective tokenizer to the model.
    device: For compatibility with different compute architectures.
    layer_id: Hard-coded layer-id to run manipulations on. Suggested at least
    2/3rd of the way through the layers -- see Gao et al. for more information on
    placement.
    """
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small", center_writing_weights=False)
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    layer_id = 9
    return model, tokenizer, device, layer_id


def prompt_summary(activations_output: pd.DataFrame, approach: str = "avg") -> pd.DataFrame:
    """Quick function to retun the prompt-level activation summaries from
    the tokens in a prompt.
    Inputs:
    - activations_output: This is a DataFrame where rows are tokens and columns are
    the activations corresponding to each SAE feature when passed through the SAE.
    - approach: type of summarization approach across rows one of either:
        - max: take the max activation per SAE feature
        - counts: number of times a token in that prompt activated on that feature
        - avg (default): average activations per SAE feature
        These summaries are then normalized across all SAE features
    Outputs:
    - a DataFrame with size [1*# of SAE features] for a prompt-level activations summary
    """
    if approach == "max":
        return activations_output.max(axis=0)/activations_output.max(axis=0).sum()
    elif approach == "counts":
        return (activations_output > 0).sum(axis=0)/(activations_output > 0).sum(axis=0).sum()
    # Returns average 
    return activations_output.sum(axis=0)/activations_output.sum(axis=0).sum()


def gather_residual_activationsgpt(model, layer: int, inputs) -> torch.tensor:
    """Helper function to quickly grab SAE activations.

    Not dependent on transformer-lens -- straightforward approach.
    Also runs the model.

    Inputs:
    - model: The LLM (generating text task) through
    - layer: Layer of LLM for which to get the activations
    - inputs: Pre-tokenized model inputs to run the function

    Outputs:
    - Returns activations per prompt as a torch.tensor across all tokens.

    """
    target_act = None

    def gather_target_act_hook(outputs, hook):
        nonlocal target_act
        target_act = outputs[0]
        return outputs

    _ = model.add_hook(f"blocks.{layer}.hook_resid_post", gather_target_act_hook)
    _ = model(inputs)
    model.reset_hooks()
    return target_act


def sentence_embeddings_maps(sample_inputs):
    """Pre-existing method to use with the sentence-transformers
    from sentence_transformer starter code:
    https://huggingface.co/sentence-transformers

    Inputs:
    - sample_inputs: a list of strings as prompts to input to the method
    Output:
    - sentence_embeddings: tensor of embeddings for all input prompts."""

    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask = attention_mask.unsqueeze(-1)
        input_mask_expanded = input_mask.expand(token_embeddings.size()).float()
        multiplication_tensors = torch.sum(token_embeddings * input_mask_expanded, 1)
        return multiplication_tensors / torch.clamp(input_mask_expanded.sum(1),
                 min=1e-9)

    sentences = sample_inputs
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    encoded_input = tokenizer(sentences, padding=True,
                            truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


def summarized_latents(prompts: List[str], tokenizer, device, model,
                    layer_id: int, sae, summary_approach='avg') -> (pd.DataFrame, float):
    
    """
    Given some prompts, return the summarized SAE latents and
    the % of features fired, also known as coverage. This gives some insights
    into the presence of ghost SAE features of the SAE and thus the respresentational power.

    Inputs:
    - prompts: list of strings of prompts to consider
        (e.g. from alignment or reference set)
    - tokenizer: Tokenizer for the model
    - device: For compute generalizability
    - model: Transformer-lens model for a generative approach
    - layer_id: Layer at which the SAE will be used
    - sae: Type of SAE to be used (default here from OpenAI)
    - summary approach: how to change token-level activations
        from SAE to prompt-level activations as desired.

    Outputs:
    - all_summary_latents: prompt-level SAE activations per input prompt.
    - coverage: % of fired SAE SAE features across all prompts considered.
    """
    all_outputs = []
    for prompt in prompts:
        tok = tokenizer.encode(prompt, return_tensors="pt").to(device)
        target_act = gather_residual_activationsgpt(model, layer_id, tok)
        lats, _ = sae.encode(target_act.to(device))
        lats = lats.cpu()
        # remove BOS token
        modify_lats = prompt_summary(pd.DataFrame(lats[1:]), summary_approach)
        all_outputs.append(modify_lats)
    all_summary_latents = pd.concat(all_outputs, axis=1).T
    coverage = (((all_summary_latents > 0).sum(axis=0)/all_summary_latents.shape[0])>0).sum()/all_summary_latents.shape[1]
    return all_summary_latents, float(coverage)


def process_D_ref(ref_prompts: List[str], tokenizer, device, model, layer_id : int):
    """
    Structuring function to format reference (not alignment)
    prompts for summarized_latents function and embeddings in the sentence-space.
    Inputs:
    - ref_prompts: This are a list of strings corresponding to the reference (not alignment) set.
    - tokenizer: Tokenizer for the model
    - device: For compute generalizability
    - model: Transformer-lens model for a generative approach
    - layer_id: Layer at which the SAE will be used

    Outputs:
    - ref_embd: Sentence-level embeddings for prompts in D_ref
    - autoencoder: Selected autoencoder: takes a while to download
    - all_summary_latents: prompt-level SAE activations per input prompt.
    - coverage: % of fired SAE SAE features across all prompts considered.
    """

    ref_embd = pd.DataFrame(sentence_embeddings_maps(ref_prompts))
    ref_embd.to_csv('Storage/D_ref_embeds.csv')
    layer_index = layer_id
    location = "resid_post_mlp"

    # From OpenAI Sparse Autoencoder --
    # download for faster speeds: https://github.com/openai/sparse_autoencoder
    # Replace with desired autoencoder:
    with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
        state_dict = torch.load(f)
        autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
        autoencoder.to(device)

    all_summary_latents, coverage = summarized_latents(ref_prompts, tokenizer,
                                        device, model, layer_id, autoencoder)
    all_summary_latents.to_csv(f'Storage/dref_latents_{layer_id}.csv')

    return ref_embd, autoencoder, all_summary_latents, coverage


def D_align_scoring(align_prompts, ref_embd, layer_id):
    """
    Process alignment prompts for only sentence-space embeddings to calculate the
    min-distance between D_ref sentence-space embeddings and D_align which
    are used to output relevance to D_align scores between 0-1 per SAE feature.

    Inputs:
    - align_prompts: List of strings in the alignment set.
    - ref_embd: Embeddings from the reference set
    - layer_id: Layer at which the SAE will be used - only for saving files

    Outputs:
    - score: relevance to D_align scores between 0-1 per SAE feature.
    """
    align_df = pd.DataFrame(sentence_embeddings_maps(align_prompts))
    align_df.to_csv('Storage/D_align_embeds.csv')
    # calculate the minmum distance between every reference embedding and the
    # alignment embeddings
    dist_df = pd.Series([min(euclidean_distances(x.to_numpy().reshape(1, -1),
                        align_df)[0]) for _, x in ref_embd.iterrows()])
    acts_ref_embd = pd.read_csv(f'Storage/dref_latents_{layer_id}.csv',
                                index_col=0, header=0)
    feature_num = np.zeros((acts_ref_embd.shape[1], 1))
    feature_denom = np.zeros((acts_ref_embd.shape[1], 1))
    dist_multiplier = np.multiply(dist_df, dist_df).T
    for i, act_summary in acts_ref_embd.iterrows():
        og = (dist_multiplier[int(i)]*act_summary).astype(float)
        feature_num += np.reshape(og, feature_denom.shape)
        feature_denom += np.reshape(act_summary, feature_denom.shape)
    # Simple transformation so higher scores are better
    scores = feature_num/(feature_denom)
    scores = scores - scores.min()
    scores = 1-scores/scores.max()
    return np.nan_to_num(scores).T


def original(activations: torch.tensor, hook, layer : int = 0, sae = None, scores: torch.tensor = None,
                    final_layer : int = None, device = None, strs:List[str] = None) -> torch.tensor:
    """
    In these functions, we use the transformer lens hooks to modify
    activations using strategies: original - no change,
    SAE - using an SAE at the layer (with no tuning),
    clamp - existing SOTA to take SAE features relevant to a topic and clamp them high
    weight - weigh preactivations scores in the SAE by SAE features and take activations
    swap  - take indices of weight and use them with the original SAE activations.

    These functions have been separated out for timing experiments.
    We also save intermediate outputs as DataFrames for later processing.

    This is the baseline approach.

    Inputs: 
    - activations: from the hook
    - hook: mandated by transformer lens
    - layer: where activations are taken from
    - sae: the autoencoder for processing activations
    - scores: the scores per SAE feature for modifying output
    - final_layer: necessary flag for processing to ensure that 
        SAE activations are only being applied at the layer if the user
        wants to save last-layer activations as well. 
    - device: For compute generalizability
    - strs: the strings corresponding to each token for easier analysis.

    Outputs: 
    - activations: the modified activations after this step (in this case, no change. )
    """
    pd.DataFrame(activations.cpu()[0],
                    index=strs).to_csv(f'Storage/orig_output_at_layer_{layer}.csv')
    return activations


def SAE(activations: torch.tensor, hook, layer : int = 0, sae = None, scores: torch.tensor = None, 
                    final_layer : int = None, device = None, strs:List[str] = None) -> torch.tensor:
    """
    See docstring for the ``original`` function due to limitations on #LOC
    This approach uses only the SAE but does no modifications.
    """
    if layer != final_layer:
        lats, info = sae.encode(activations.to(device))
        sae_activations = sae.decode(lats, info)
        pd.DataFrame(lats.cpu()[0],
            index=strs).to_csv(f'Storage/sae_latents_at_layer_{layer}.csv')
        return sae_activations
    pd.DataFrame(activations.cpu()[0],
    index=strs).to_csv(f'Storage/sae_output_at_layer_{layer}.csv')
    return activations


def clamp(activations: torch.tensor, hook, layer : int = 0, sae = None, scores: torch.tensor = None, 
                    final_layer : int = None, device = None, strs:List[str] = None) -> torch.tensor:
    """
    See docstring for the ``original`` function due to limitations on #LOC
    This approach uses the SAE and artifically sets high-scoring SAE features high.
    """
    if layer != final_layer: 
        clamp_lats, info = sae.encode(activations)
        top_features = torch.topk(scores, 5).indices.cpu().tolist()
        clamp_lats[:, :, top_features]  = clamp_lats[:, :, top_features]+1*10
        clamp_activations = sae.decode(clamp_lats, info) 
        pd.DataFrame(clamp_lats.cpu()[0],
                index=strs).to_csv(f'Storage/clamp_latents_at_layer_{layer}.csv')
        return clamp_activations
    pd.DataFrame(activations.cpu()[0],
                index=strs).to_csv(f'Storage/clamp_output_at_layer_{layer}.csv')
    return activations


def weight(activations: torch.tensor, hook, layer : int = 0, sae = None, scores: torch.tensor = None,
                    final_layer : int = None, device = None, strs:List[str] = None) -> torch.tensor:
    """
    See docstring for the ``original`` function due to limitations on #LOC
    
    This approach weighs preactivations scores in the SAE by SAE features and take activations

    """

    if layer != final_layer: 
        x, info2 = sae.preprocess(activations)
        preactivations = sae.encode_pre_act(x)
        activations_lats = torch.multiply(preactivations, scores)
        mult_acts = sae.activation(activations_lats.type(torch.FloatTensor))
        info2.update(dict(mu=scores.mean(), std=scores.mean()**2))
        weight_activations = sae.decode(mult_acts, info2)
        pd.DataFrame(mult_acts.cpu()[0],
                    index=strs).to_csv(f'Storage/weight_latents_at_layer_{layer}.csv')
        return weight_activations
    pd.DataFrame(activations.cpu()[0],
                index=strs).to_csv(f'Storage/weight_output_at_layer_{layer}.csv')
    return activations


def swap(activations: torch.tensor, hook, layer : int = 0, sae = None, scores: torch.tensor = None,
                    final_layer : int = None, device = None, strs:List[str] = None) -> torch.tensor:
    """
    This is our suggested method! 
    See docstring for the ``original`` function due to limitations on #LOC 
    This approach takes the indices of weight and use them with the original SAE activations.
    """
    if layer != final_layer:
        x, info2 = sae.preprocess(activations)
        preactivations = sae.encode_pre_act(x) 
        activations_lats = torch.multiply(preactivations, scores).type(torch.FloatTensor)
        mult_acts = sae.activation(activations_lats)
        mask = torch.zeros(1, preactivations.size(1), preactivations.size(2))
        mask[torch.nonzero(mult_acts, as_tuple=True)] = 1
        swap_lats = torch.mul(preactivations, mask.to(device))
        swap_activations = sae.decode(swap_lats, info2)
        pd.DataFrame(swap_lats.cpu()[0],
                index=strs).to_csv(f'Storage/swap_latents_at_layer_{layer}.csv')
        return swap_activations
    pd.DataFrame(activations.cpu()[0],
                index=strs).to_csv(f'Storage/swap_output_at_layer_{layer}.csv')
    return activations


# A dictionary to map modification approach titles to functions.
method_func = { 'orig' : original,
                'SAE' : SAE,
                'clamp' : clamp,
                'weight' : weight,
                'swap' : swap}


def manipulate_d_align(prompt: str, model, sae, scores, layer_id:int,
                 device, method='swap', final_layer=11):
    """
    Function to structure the manipulation approach call.

    Additional layers can be added to fwd_hooks in the last list item
    #Structure from 
    # https://www.lesswrong.com/posts/hnzHrdqn3nrjveayv/how-to-transformer-mechanistic-interpretability-in-50-lines

    Inputs: 
    - prompt: input prompt to process using a manipulation method (see method_func dictionary)
    - model: HookedTransformer LLM 
    - sae: the autoencoder for processing activations
    - scores: the scores per SAE feature for modifying output
    - layer_id: where the activations are pulled from
    - device: For compute generalizability
    - method: manipulation method (see method_func dictionary)
    - final_layer: for GPT2 this is 11.

    Outputs:
    - logits: outputs from generative logits

    """
    scores = torch.tensor(scores).to(device)
    strs = model.to_str_tokens(prompt)
    fwd_hooks = [(f"blocks.{x}.hook_resid_post", 
                partial(method_func[method], 
                            layer = x, 
                            sae = sae, 
                            scores = scores,
                            final_layer = final_layer,
                            device = device,  
                            strs = strs)) for x in [layer_id]]
    logits = model.run_with_hooks(prompt,fwd_hooks=fwd_hooks)
    return logits



def generate_mainpulated_output(prompts, model,sae, scores, layer_id, 
                    device, method='swap', max_new_tokens=10):
    """
    Function to structure the manipulation approach call.

    Additional layers can be added to fwd_hooks in the last list item
    #Structure from 
    # https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Patchscopes_Generation_Demo.ipynb

    Inputs: 
    - prompts: input prompts to process using different methods
    - model: HookedTransformer LLM
    - sae: the autoencoder for processing activations
    - scores: the scores per SAE feature for modifying output
    - layer_id: where the activations are pulled from
    - device: For compute generalizability
    - method: manipulation method (see method_func dictionary)
    - max_new_tokens: # of tokens to generate

    Outputs: 
    - all_responses: generated text from model.

    """
    #Simple searching approach for compute reasons
    all_responses = []
    for prompt in prompts: 
        temp_prompts = prompt
        input_tokens = model.to_tokens(temp_prompts)
        for _ in range(max_new_tokens):
            logits = manipulate_d_align(
                temp_prompts, 
                model, 
                sae, 
                scores, 
                layer_id, 
                device,
                method=method
            )
            p = random.choice(
                                torch.topk(logits[:, -1, :].flatten(), 5)
                                .indices.cpu()
                                .tolist()
                            )
            next_tok = torch.Tensor([p]).to(torch.long).to(device)
            input_tokens = torch.cat((input_tokens,
                             next_tok.view(input_tokens.size(0), 1)), dim=1)
            temp_prompts = model.to_string(input_tokens)
        all_responses.append(model.to_string(input_tokens)[0])
    return all_responses


def fine_tuning_demo(prompt, device):
    """
    Comparison approach for the demo.

    Inputs:
    prompt: prompt for fine-tuning
    device: for compute generalization.
    Output:
    Generated output - prints time for inference as well.
    """
    start_ft = time.time()
    output_dir = "./Storage/model_save/"
    model = GPT2LMHeadModel.from_pretrained(output_dir)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model.eval()
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)
    sample_outputs = model.generate(
        generated,
        # bos_token_id=random.randint(1,30000),
        do_sample=True,
        top_k=5,
        min_new_tokens=64,
        max_new_tokens=64,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    print("Fine Tuning")
    print("Timing: ", time.time() - start_ft)
    for i, sample_output in enumerate(sample_outputs.sequences):
        print(
            "{}: {}\n\n".format(
                i, tokenizer.decode(sample_output, skip_special_tokens=True)
            )
        )
    # print(f"Perplexity:{torch.exp(torch.cat(sample_outputs.logits[-64:]).mean()).cpu().numpy()}")


def swap_demo(prompt, model, tokenizer, device):
    """
    Demonstration of approach for demo using swap and original methods

    Inputs:
    - prompt: prompt for fine-tuning
    - model: HookedTransformer model from processing
    - device: for compute generalization.
    Output:
    - Generated output - prints time for inference as well
    """
    max_new_tokens = 10
    scores = pd.read_csv("./Storage/medical_scores.csv", index_col=0, header=0).iloc[:, 0]
    layer_index = 0  # corresponds to scores file
    location = "resid_post_mlp"
    with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
        state_dict = torch.load(f)
        autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
        autoencoder.to(device)
    for method in ['swap', 'orig']:
        temp_prompts = prompt
        start = time.time()
        random.seed(0)
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
        print(f"{method} method timing:", time.time() - start)
        print(temp_prompts)

#Aux functions for processing Gemma Code as well as large prompts. 

def large_prompt_process(prompt, model, tokenizer, sae, layer, device, typ='gpt'): 
    """Special handling functions for larger prompts for both GPT and Gemma. 
    
    Inputs: 
    - prompt: text over which to create latents
    - model: LLM under consideration 
    - tokenizer: respective tokenizer
    - sae: Sparse auto-encoder under consideration
    - layer: layer to take latents 
    - type: GPT or Gemma 

    Outputs: Respective processed latents 
    """
    layer_id = spec['layer']
    tok = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if tok.size(1) > 1024:
        cat_list = []
        for x in range(tok.size(1)//1024):
            if spec['typ']!= 'gpt':
                target_act = gather_residual_activations(model, layer_id, tok[:, x*1024:(x+1)*1024])
                lats=  sae.encode(target_act.to(device))[0]
            else: 
                target_act = gather_residual_activationsgpt(model, layer_id, tok[:, x*1024:(x+1)*1024])
                lats, _ =  sae.encode(target_act.to(device))
            cat_list.append(lats)
        x = x+1
        if spec['typ'] != 'gpt':
            target_act = gather_residual_activations(model, layer_id, tok[:, x*1024:])
            lats=  sae.encode(target_act.to(device))[0]
        else: 
            target_act = gather_residual_activationsgpt(model, layer_id, tok[:, x*1024:])
            lats, _ =  sae.encode(target_act.to(device))
        cat_list.append(lats)
        lats = torch.cat(cat_list)
    else:
        if spec['typ'] != 'gpt':
            target_act = gather_residual_activations(model, layer_id, tok)
            lats=  sae.encode(target_act.to(device))[0]
        else:
            target_act = gather_residual_activationsgpt(model, layer_id, tok)
            lats, _ =  sae.encode(target_act.to(device))
    lats = lats.cpu()
    output = pd.DataFrame(lats[1:])
    return output

def gather_residual_activations(model, target_layer, inputs):
    """More generic approach to gathering activations (for Gemma).
    See specs for gather_residual_activationsgpt method.
    Output is slightly different format."""
    target_act = None
    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act 
        target_act = outputs[0]
        return outputs
    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model.forward(inputs)
    handle.remove()
    return target_act

# Datatset generation (for reference)

def generate_files(path_remote, access_token, cache_dir):
    """ A quick way to pull relevant datasets from HuggingFace and create data files structure from them 
    Inputs: 
    - path_remote: for organizational purposes
    - access_token: to prevent secrets leakage
    - cache_dir: to help manage large files from HF 
    Outputs: 
    None
    """
    for typ_file in ["test", "train"]:
        df = pd.DataFrame()  # filename,
        random.seed(0)
        HfFileSystem()
        if typ_file == "train":
            goal = 20
        else:
            goal = 20
        ds1 = load_dataset(
            "bigscience/P3",
            "amazon_polarity_Is_this_product_review_positive",
            split=typ_file,
            cache_dir=cache_dir,
            token=access_token,
        )["inputs_pretokenized"]
        if typ_file == "train":
            ds2 = load_dataset(
                "gamino/wiki_medical_terms",
                split=typ_file,
                cache_dir=cache_dir,
                token=access_token,
            )["page_text"]
        else:
            ds2 = load_dataset(
                "qiaojin/PubMedQA",
                "pqa_labeled",
                split="train",
                cache_dir=cache_dir,
                token=access_token,
            )["question"]
        for ds, fname in zip([ds1, ds2], ["amz", "med"]):
            if not os.path.exists(f"{path_remote}/{fname}"):
                os.mkdir(f"{path_remote}/{fname}")
            random.shuffle(ds)
            sel = pd.DataFrame()
            sel["prompts"] = ds[: min(len(ds), goal)]
            sel["prompts"] = (
                sel["prompts"]
                .str.replace("\n", "")
                .str.replace(",", "")
                .str.replace("Title:", "")
                .str.replace("Review:", "")
                .str.replace("Answer:", "")
            )
            if typ_file == "train":
                sel.to_csv(f"{path_remote}/{fname}/train_20.csv")
                sel.sample(6).to_csv(f"{path_remote}/{fname}/train_6.csv")
            else:
                sel.to_csv(f"{path_remote}/{fname}/test.csv")
        prompt_template = "{question}:{answer}"  # only keep the sycophantic answer } :
        MWEData = list[dict[str, str]]

        def make_pos_neg_pair(mwe_data: MWEData, fname) -> tuple[str, str]:
            """Creates a (positive, negative) pair for getting contrastive activations.
            Processing from 
            https://colab.research.google.com/github/steering-vectors/steering-vectors/blob/main/examples/caa_sycophancy.ipynb#scrollTo=sH6zCVxVVqa8
             See this link for additional details on function inputs/outputs 
            """

            answer_a = mwe_data["question"].split("\n")[-2][5:]
            answer_b = mwe_data["question"].split("\n")[-1][5:]
            question = " ".join(mwe_data["question"].split("\n")[:-2])
            answer_correct = mwe_data["answer_matching_behavior"]
            if answer_correct == "(A)":
                final_answer = answer_a
                other_answer = answer_b
            elif answer_correct == "(B)":
                final_answer = answer_b
                other_answer = answer_a
            else:
                asdfasdf
            pos = prompt_template.format(
                question=question, answer=final_answer
            ).replace(",", "")
            prompt_template.format(question=question, answer=other_answer)
            return [
                {"prompts": pos},
            ]  # {'prompts':neg, 'filename':f"{fname}_neg"}

        for cont_file in ["sycophancy"]:
            url = f"https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/test/{cont_file}/test_dataset_ab.json"
            if typ_file == "train":
                url = f"https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/generate/{cont_file}/generate_dataset.json"
            wget.download(url, f"generate_dataset_{cont_file}_{typ_file}.json")
            train_data: list[MWEData] = json.load(
                open(f"generate_dataset_{cont_file}_{typ_file}.json", "r")
            )
            train_data = train_data[
                : min(len(train_data), goal)
            ]  # half due to contrastive set
            nested_df = [
                make_pos_neg_pair(mwe_data, cont_file) for mwe_data in train_data
            ]  # only pos pair for now

            def flatten(xss):
                return [x for xs in xss for x in xs]

            df = pd.DataFrame(flatten(nested_df))
            if not os.path.exists(f"{path_remote}/{cont_file}"):
                os.mkdir(f"{path_remote}/{cont_file}")
            if typ_file == "train":
                df.to_csv(f"{path_remote}/{cont_file}/train_20.csv")
                df.sample(6).to_csv(f"{path_remote}/{cont_file}/train_6.csv")
            else:
                df.to_csv(f"{path_remote}/{cont_file}/test.csv")
        df = pd.read_csv("CCC-deploy/Datasets/shoes.csv", header=0)
        cont_files = "shoes"
        if typ_file == "train":
            df.prompts.iloc[:20].str.replace("\n", "").str.replace(",", "").to_csv(
                f"{path_remote}/{cont_files}/train_20.csv"
            )
            df.prompts.iloc[20:26].str.replace("\n", "").str.replace(",", "").to_csv(
                f"{path_remote}/{cont_files}/train_6.csv"
            )
        else:
            df.prompts.iloc[26:].str.replace("\n", "").str.replace(",", "").to_csv(
                f"{path_remote}/{cont_files}/test.csv"
            )

