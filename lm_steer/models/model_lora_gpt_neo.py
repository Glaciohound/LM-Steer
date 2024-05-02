import torch
import torch.nn as nn
from transformers import pipeline
from peft import LoraConfig, get_peft_model

from lm_steer.utils import set_seed


class LORA_GPTNeoModel(nn.Module):
    def __init__(self, model_name, rank, epsilon):
        super().__init__()
        self.generator = pipeline('text-generation',
                                  model=model_name.replace("lora-", ""))
        self.tokenizer = self.generator.tokenizer
        model = self.generator.model
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        config = LoraConfig(
            r=rank,
            lora_alpha=epsilon,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=[],
        )
        self.model = get_peft_model(model, config)
        self.generator.model = self.model
        self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, steer_values):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids)
        return output

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
