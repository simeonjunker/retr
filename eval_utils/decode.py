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


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


def greedy(model, image, tokenizer, start_token, max_pos_embeddings):

    caption, cap_mask = create_caption_and_mask(
        start_token, max_pos_embeddings)

    with torch.no_grad():
        model.eval()

        for i in range(max_pos_embeddings - 1):
            predictions = model(image, caption, cap_mask)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                break

            caption[:, i+1] = predicted_id[0]
            cap_mask[:, i+1] = False

    return tokenizer.decode(caption[0], skip_special_tokens=True)
