# core/evaluation_utils.py
from dependencies import cosine, np, roc_curve, torch
from config import EER_P_TARGET, EER_C_MISS, EER_C_FA


def all_metrics(similarities, labels, p_target=0.05, c_miss=1, c_fa=1):
    # Calculate False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    fnr = 1 - tpr  # False Negative Rate

    # Find EER (Equal Error Rate) and its thresholds
    eer_indices = np.where(np.abs(fnr - fpr) == np.min(np.abs(fnr - fpr)))[0]
    eer = fpr[eer_indices[0]]
    eer_threshold_more = thresholds[eer_indices[-1]]  # More strict (higher threshold)
    eer_threshold_less = thresholds[eer_indices[0]]  # Less strict (lower threshold)
    eer_threshold_wst = np.mean(thresholds[eer_indices])  # Average of all EER thresholds

    # Calculate minDCF (Minimum Detection Cost Function)
    c_det = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_dcf_indices = np.where(c_det == np.min(c_det))[0]
    min_dcf = c_det[min_dcf_indices[0]]
    min_dcf_threshold_more = thresholds[min_dcf_indices[-1]]  # More strict (higher threshold)
    min_dcf_threshold_less = thresholds[min_dcf_indices[0]]  # Less strict (lower threshold)
    min_dcf_threshold_wst = np.mean(thresholds[min_dcf_indices])  # Average of all minDCF thresholds

    # Create and return the results dictionary
    results = {
        "fpr": fpr.tolist(),  # Convert arrays to lists for better usability in JSON, if needed
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "fnr": fnr.tolist(),
        "eer": eer,
        "eer_threshold_more": eer_threshold_more,
        "eer_threshold_less": eer_threshold_less,
        "eer_threshold_wst": eer_threshold_wst,
        "min_dcf": min_dcf,
        "min_dcf_threshold_more": min_dcf_threshold_more,
        "min_dcf_threshold_less": min_dcf_threshold_less,
        "min_dcf_threshold_wst": min_dcf_threshold_wst,
    }

    return results

# Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR) at different thresholds
def compute_metrics(similarities, labels, p_target=0.05, c_miss=1, c_fa=1, strictness="wst"):
    # Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR) at different thresholds
    results = all_metrics(similarities, labels, p_target, c_miss, c_fa, strictness)
    eer = results["eer"]
    
    if strictness == "more":
        eer_threshold = results["eer_threshold_more"]  # More strict (higher threshold)
    elif strictness == "less":
        eer_threshold = results["eer_threshold_less"]  # Less strict (lower threshold)
    else:
        eer_threshold = results["eer_threshold_wst"]  # Average of all EER thresholds

    min_dcf = results["min_dcf"]
    
    if strictness == "more":
        min_dcf_threshold = results["min_dcf_threshold_more"]  # More strict (higher threshold)
    elif strictness == "less":
        min_dcf_threshold = results["min_dcf_threshold_less"]  # Less strict (lower threshold)
    else:
        min_dcf_threshold = results["min_dcf_threshold_wst"]  # Average of all minDCF thresholds
    
    return eer, min_dcf, eer_threshold, min_dcf_threshold

def evaluate_live(model, reference_embedding, test_embedding, threshold):
    # Detach the tensor from the computation graph and convert it to a NumPy array
    if isinstance(test_embedding, torch.Tensor):
        test_embedding = test_embedding.detach().cpu().numpy()
    if isinstance(reference_embedding, torch.Tensor):
        reference_embedding = reference_embedding.detach().cpu().numpy()

    # Flatten embeddings to 1-D if necessary
    test_embedding = test_embedding.flatten()
    reference_embedding = reference_embedding.flatten()
    
    # Calculate cosine similarity
    similarity = 1 - cosine(reference_embedding, test_embedding)
    return similarity >= threshold
