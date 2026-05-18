from functions import (
    ModelInitial,
    Gradients,
    FisherInfo,
    Pruning,
    Evaluation
)

# =========================================================
# LOAD MODEL
# =========================================================

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

obj = ModelInitial(model_name)

# =========================================================
# EVALUATION BEFORE PRUNING
# =========================================================

print("\n================ BEFORE PRUNING ================\n")

before_eval = Evaluation(obj)

# =========================================================
# SAMPLE INPUT
# =========================================================

text = [
    "Large language models are useful for many AI tasks."
]

# =========================================================
# COMPUTE GRADIENTS
# =========================================================

grad_obj = Gradients(obj, text)

# =========================================================
# COMPUTE FISHER INFORMATION
# =========================================================

fisher_obj = FisherInfo(
    grad_obj.gradients,
    "fine"
)

# =========================================================
# APPLY FINE-GRAINED PRUNING
# =========================================================

Pruning(
    fisher_obj.fisher_info,
    obj,
    prune_percent=0.20
)

print("\n20% Fine-Grained Pruning Completed\n")

# =========================================================
# EVALUATION AFTER PRUNING
# =========================================================

print("\n================ AFTER PRUNING ================\n")

after_eval = Evaluation(obj)