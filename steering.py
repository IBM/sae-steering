#Ananya Joshi
#Sep 18 2025
#Examples of Steering functions 
import pandas as pd
from steering_utils import (
    process_D_ref,
    D_align_scoring,
    generate_mainpulated_output,
    setup,
    fine_tuning_demo,
    swap_demo
)

def generated_ouptuts(): 
    """
    This module covers both training and inference steps for this approach. 
    """
    # Set Variables
    model, tokenizer, device, layer_id = setup()

    # Load Reference Datasets
    D_ref = list(pd.read_csv("Storage/d_ref_prompts.csv",
                        index_col=0, header=0)["0"])
    D_align = list(pd.read_csv("Storage/d_align_prompts.csv",
                        index_col=0, header=0)["0"])
    D_align_test = list(pd.read_csv("Storage/d_align_test_prompts.csv",
                        index_col=0, header=0)["0"])

    # Generate SAE Score Weights for D_align topics
    ref_dist, autoencoder, latents, coverage = process_D_ref(
        D_ref, tokenizer, device, model, layer_id
    )
    scores = D_align_scoring(D_align, ref_dist, layer_id)

    # Return Generated Outputs
    generated_output = generate_mainpulated_output(
        D_align_test,
        model,
        autoencoder,
        scores,
        layer_id,
        device,
        method="swap",
    )
    print(generated_output)


def demo(): 
    """
    An example of how this approach works using scores in the medical domain. 
    """
    model, tokenizer, device, layer_id = setup()
    while True:
        prompt = input("Your prompt to tune to medical: ")
        fine_tuning_demo(prompt, device)
        swap_demo(prompt, model, tokenizer, device)

# generated_ouptuts()

# May return results that are harmful or inappropriate - please use with discretion. 
if __name__ == "__main__":
    demo()