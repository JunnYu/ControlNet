import paddle
import paddle.nn as nn

from paddlenlp.transformers import CLIPTokenizer,CLIPTextConfig
from ...clip import  CLIPTextModel


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - paddle.bernoulli(paddle.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * paddle.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = paddle.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden",
        "penultimate",
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77,
                 freeze=True, layer="last", layer_idx=None, 
                 from_hf_hub = False,
                 cache_dir=None,
                 **kwargs):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version, from_hf_hub=False, cache_dir=cache_dir)
        self.tokenizer.pad_token_id = 0
        config = CLIPTextConfig.from_pretrained(version, from_hf_hub=from_hf_hub, cache_dir=cache_dir)

        self.max_length = min(max_length, self.tokenizer.model_max_length)

        self.layer = layer
        
        self.layer_idx = layer_idx
        
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= config.num_hidden_layers

        # update config
        if layer == "penultimate":
            config.num_hidden_layers = config.num_hidden_layers - 1
            # assert config.num_hidden_layers == 23

        if not from_hf_hub:
            self.transformer = CLIPTextModel(config)
        else:
            self.transformer = CLIPTextModel.from_pretrained(version, config=config, from_hf_hub=from_hf_hub, cache_dir=cache_dir)

        if freeze:
            self.freeze()
    def freeze(self):
        self.transformer = self.transformer.eval()
        self.eval()
        for param in self.parameters():
            param.stop_gradient = True

    def forward(self, text):
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pd").input_ids
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer in ["last", "penultimate"]:
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        elif self.layer == "hidden":
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)
