from collections import Counter
import torch

# CHATGPT for the implementation of the metric
def f1_score(pred_toks, true_toks):
    """
    Function to calculate F1-score
    """
    common = Counter(pred_toks) & Counter(true_toks)  # Find common tokens
    num_common = sum(common.values())  # Count how many tokens are in common
    if num_common == 0:
        return 0
    precision = num_common / len(pred_toks)
    recall = num_common / len(true_toks)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_metric(outputs, input_ids, start_positions, end_positions):
    """
    Computes start and end predictions and calculates F1 score.
    return: average f1_score of a batch
    """
    start_preds = torch.argmax(outputs.start_logits, dim=1)
    end_preds = torch.argmax(outputs.end_logits, dim=1)

    total_f1 = 0
    batch_size = input_ids.size(0)

    for i in range(batch_size):
        pred_tokens = input_ids[i][start_preds[i]:end_preds[i] + 1].tolist()
        true_tokens = input_ids[i][start_positions[i]:end_positions[i] + 1].tolist()
        total_f1 += f1_score(pred_tokens, true_tokens)

    average_f1_score = total_f1 / batch_size
    return average_f1_score