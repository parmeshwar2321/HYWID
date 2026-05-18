from functions import (
    model_initial,
    Gradients,
    FisherInfo,
    Pruning,
    Evaluation,
    AttentionFusion
)

import torch.nn.functional as F


# LOAD MODEL

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

obj = model_initial(model_name)

# BEFORE PRUNING

print("\n========== BEFORE PRUNING ==========\n")

before_eval = Evaluation(obj)

# INPUT TEXT

text = [
    "Large language models are useful for many AI tasks."
]

# COMPUTE GRADIENTS

grad_obj = Gradients(obj, text)

# FINE IMPORTANCE

fine_fisher = FisherInfo(
    grad_obj.gradients,
    "fine"
)

# COARSE IMPORTANCE

coarse_fisher = FisherInfo(
    grad_obj.gradients,
    "coarse"
)

# ATTENTION FUSION MODULE

fusion_module = AttentionFusion(
    d_model=128
)

# HYBRID IMPORTANCE

hybrid_fisher = {}

for name in fine_fisher.fisher_info:

    fine = fine_fisher.fisher_info[name]

    coarse = coarse_fisher.fisher_info[name]

    
    # ONLY MATRIX PARAMETERS
    

    if fine.dim() < 2:

        hybrid_fisher[name] = fine

        continue

    
    # FLATTEN
    

    fine_vector = fine.flatten()

    coarse_vector = coarse.flatten()

    original_size = fine_vector.numel()

    
    # PAD
   

    pad = (128 - original_size % 128) % 128

    if pad > 0:

        fine_vector = F.pad(
            fine_vector,
            (0, pad)
        )

        coarse_vector = F.pad(
            coarse_vector,
            (0, pad)
        )


    # RESHAPE

    fine_vector = fine_vector.view(-1, 128)

    coarse_vector = coarse_vector.view(-1, 128)

    
    # FUSION
    

    fused = fusion_module(
        fine_vector,
        coarse_vector
    )

    
    # RESTORE SHAPE

    fused = fused.flatten()[:original_size]

    fused = fused.view_as(fine)

    hybrid_fisher[name] = fused



Pruning(
    hybrid_fisher,
    obj,
    prune_percent=0.20
)

print("\n20% HyWIA-style pruning applied.\n")



print("\n========== AFTER PRUNING ==========\n")

after_eval = Evaluation(obj)