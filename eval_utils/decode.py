import torch
from transformers import BertTokenizer
from PIL import Image


def prepare_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    return tokenizer, start_token, end_token


def load_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def create_caption_and_mask(start_token, max_length, batch_size=1):
    caption_template = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask_template = torch.ones((batch_size, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


def greedy_single(model, image, tokenizer, start_token, end_token, max_pos_embeddings):
    """greedy decoding for a single image"""

    caption, cap_mask = create_caption_and_mask(
        start_token, max_pos_embeddings)

    with torch.no_grad():
        model.eval()

        for i in range(max_pos_embeddings - 1):
            predictions = model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == end_token:
                break

            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False

    return tokenizer.decode(caption[0], skip_special_tokens=True)


def greedy(samples, model, max_len=20, device="auto", bos_token=1, eos_token=2):
    """greedy decoding for a batch of samples"""

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    caption, cap_mask = create_caption_and_mask(
        bos_token, max_len, samples[0].shape[0])

    samples = [s.to(device) for s in samples]
    caption = caption.to(device)
    cap_mask = cap_mask.to(device)

    finished = torch.zeros(caption.shape[0], dtype=bool, device=device)

    for i in range(max_len - 1):
        predictions = model(*samples, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        is_eos = predicted_id == eos_token
        finished = torch.logical_or(is_eos, finished)
        if all(finished):
            return caption

        caption[:, i + 1] = predicted_id
        cap_mask[:, i + 1] = False

    return caption


def prune_cap_ids(idx_seqs, clean=True, pad_token=0, bos_token=1, eos_token=2):
    """cut off index sequences; optionally clean from <PAD>, <BOS> and <EOS> tokens"""
    results = []

    for seq in idx_seqs:
        pruned_seq = []

        for idx in seq:
            pruned_seq.append(idx)
            if idx == eos_token:
                break

        if clean:
            pruned_seq = [i for i in pruned_seq if i not in [pad_token, bos_token, eos_token]]

        results.append(pruned_seq)

    return results


def idx2sents(idx_seqs, tokenizer, skip_special_tokens=True):
    """convert sequences of word indices to strings"""
    return tokenizer.batch_decode(
        idx_seqs, 
        skip_special_tokens=skip_special_tokens
    )


def greedy_decoding(samples, model, tokenizer, max_len=20, clean=True, pad_token=0, bos_token=1, eos_token=2, device='auto'):
    """wrapper for greedy decoding"""

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    caption_idx = greedy(samples, model, max_len=max_len, bos_token=bos_token, eos_token=eos_token, device=device)
    caption_idx = caption_idx.cpu().detach().numpy().tolist()

    pruned_caption_idx = prune_cap_ids(
        caption_idx, clean=clean,
        pad_token=pad_token, bos_token=bos_token, eos_token=eos_token
    )

    sents = idx2sents(pruned_caption_idx, tokenizer)

    return sents


def greedy_with_att(model, sample, tokenizer, start_token=1, end_token=2, max_pos_embeddings=128, return_raw=True, device='auto'):
    """greedy decoding for a single image, outputs attention"""

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    caption, cap_mask = create_caption_and_mask(
        start_token, max_pos_embeddings)
    
    sample = [s.to(device) for s in sample]
    caption = caption.to(device)
    cap_mask = cap_mask.to(device)
    
    atts = []

    with torch.no_grad():
        model.eval()

        for i in range(max_pos_embeddings - 1):
            predictions, att = model(*sample, caption, cap_mask, return_attention=True)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
                        
            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False
            atts.append(att)

            if predicted_id[0] == end_token:
                break

    token_ids = caption[0][~cap_mask[0]]
    token_ids = token_ids[1:]  # exclude first token (BOS)

    # return raw sequence + atts
    if return_raw:
        return token_ids, atts
    # return decoded sequence + atts
    return tokenizer.decode(token_ids, skip_special_tokens=True), atts