import torch

from models.text_encoder import PromptTextEncoder, SimpleTokenizer


def test_simple_tokenizer_shape():
    tokenizer = SimpleTokenizer(vocab_size=128, max_length=8)

    token_ids = tokenizer(["studio portrait", ""])

    assert token_ids.shape == (2, 8)
    assert token_ids[1].sum() == 0


def test_prompt_text_encoder_output_shape():
    tokenizer = SimpleTokenizer(vocab_size=128, max_length=8)
    encoder = PromptTextEncoder(vocab_size=128, max_length=8, embed_dim=32, num_heads=4, num_layers=1)

    token_ids = tokenizer(["studio portrait"])
    output = encoder(token_ids)

    assert output.shape == (1, 32)
    assert not torch.isnan(output).any()
