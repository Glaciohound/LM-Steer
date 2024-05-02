import torch
import torch.nn as nn
from transformers import pipeline

from .model_utils import Hack_no_grad
from lm_steer.utils import set_seed


class EmbeddingTuning_GPTNeoModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.generator = pipeline(
            'text-generation',
            model=model_name.replace("embedding_tuning-", ""))
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.transformer = Hack_no_grad(self.model.transformer)

    def forward(self, input_ids, attention_mask, steer_values):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids)
        return output

    def parameters(self):
        return [self.model.lm_head.weight]

    def state_dict(self):
        return self.model.lm_head.state_dict()

    def load_state_dict(self, state_dict):
        self.model.lm_head.load_state_dict(state_dict)

    def to_device(self, device):
        self.generator.device = device
        self.model.to(device)
        self.device = device

    def regularization_term(self):
        return torch.tensor(0)

    def generate(self, prompt, steer_values, min_length=20, max_length=100,
                 seed=None, num_beams=1, num_beam_groups=1, do_sample=True,
                 temperature=1, top_p=1):
        if seed is not None:
            set_seed(seed)
        with torch.no_grad():
            text = self.generator(
                prompt, num_beams=num_beams, num_beam_groups=num_beam_groups,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
                min_length=min_length, max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text = text[0]["generated_text"]
        return text
