import torch
import torch.nn.functional as F


def mmi(pos_predictions, neg_predictions, targets):
    """
    Maximum Mutual Information (MMI) Loss, cf. Mao et al 2016.

    Args:
        pos_predictions (torch.tensor): batched logits for target region
        neg_predictions (list[torch.tensor]): list of batched logits for negative regions
        targets (torch.tensor): batched token idx for target sequence

    Returns:
        torch.tensor: loss value
    """
    # preprocess target tensor (one-hot for filtering relevant probs)
    onehot_target = F.one_hot(
        targets, num_classes=pos_predictions.shape[1]
    ).permute(0,2,1)
    
    # combine predictions for target and negative regions
    all_predictions = [pos_predictions] + neg_predictions
    # log probs from logits
    all_log_predictions = [
        F.log_softmax(p, dim=1) for p in all_predictions
    ]
    
    # compute probabilities for target sequence tokens
    target_seq_probs = torch.zeros(len(all_predictions))
    for i, n in enumerate(all_log_predictions):
        target_seq_probs[i] = torch.sum(n * onehot_target)

    # select probability for target region and total probability for all regions
    # (numerator and denominator in formula)
    pos_prob = target_seq_probs[0]
    total_prob = torch.logsumexp(target_seq_probs, 0)
    
    return - torch.log(
        torch.exp(pos_prob) / torch.exp(total_prob)
    )


def mmi_mm(pos_predictions, neg_predictions, targets, M=0):
    """
    Max-Margin Maximum Mutual Information (MMI-MM) Loss, cf. Mao et al 2016.

    Args:
        pos_predictions (torch.tensor): batched logits for target region
        neg_predictions (torch.tensor): batched logits for (single) negative region
        targets (torch.tensor): batched token idx for target sequence

    Returns:
        torch.tensor: loss value
    """
    # preprocess target tensor (one-hot for filtering relevant probs)
    onehot_target = F.one_hot(
        targets, num_classes=pos_predictions.shape[1]
    ).permute(0,2,1)
    
    # log probs from logits
    pos_predictions = F.log_softmax(pos_predictions, dim=1)
    neg_predictions = F.log_softmax(neg_predictions, dim=1)

    # probabilities for target sequence tokens
    pos_prob = torch.sum(pos_predictions * onehot_target)
    neg_prob = torch.sum(neg_predictions * onehot_target)

    # Comparison between target region and negative region
    return torch.max(
        torch.tensor(0), 
        M - pos_prob + neg_prob
    )


def mmi_mm_with_ce(pos_predictions, neg_predictions, targets, M=0):
    """
    Max-Margin Maximum Mutual Information (MMI-MM) Loss, cf. Mao et al 2016.; 
    summed log probabilities are replaced with cross entropy scores

    Args:
        pos_predictions (torch.tensor): batched logits for target region
        neg_predictions (torch.tensor): batched logits for (single) negative region
        targets (torch.tensor): batched token idx for target sequence

    Returns:
        torch.tensor: loss value
    """
    pos_CE = F.cross_entropy(pos_predictions, targets)
    neg_CE = F.cross_entropy(neg_predictions, targets)
    return torch.max(
        torch.tensor(0), 
        M + pos_CE - neg_CE
    )