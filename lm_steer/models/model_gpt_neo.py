import torch
from transformers import pipeline

from .model_utils import Hack_no_grad
from .steers import Projected_Adaptor
from .model_base import LMSteerBase


class Switching_GPTNeoModel(LMSteerBase):
    def __init__(self, model_name, adapted_component, adaptor_class,
                 num_steers, rank, epsilon, init_var,
                 low_resource_mode):
        super().__init__()
        self.adapted_component = adapted_component
        self.generator = pipeline('text-generation', model=model_name)
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.init_var = init_var
        self.num_steers = num_steers
        self.device = torch.device("cpu")
        embed_dim = self.model.lm_head.weight.shape[1]
        vocab_size = self.model.lm_head.weight.shape[0]

        for _param in self.model.parameters():
            _param.requires_grad_(False)

        if adapted_component == "final_layer":
            self.model.transformer = Hack_no_grad(self.model.transformer)
            self.steer = Projected_Adaptor(
                self.model.lm_head, adaptor_class, num_steers, embed_dim,
                vocab_size, rank, epsilon, init_var, "output")
            self.model.set_output_embeddings(self.steer)
        elif adapted_component == "input_embedding":
            self.steer = Projected_Adaptor(
                self.model.transformer.wte, adaptor_class, num_steers,
                embed_dim, vocab_size, rank, epsilon, init_var, "input")
            self.model.transformer.set_input_embeddings(self.steer)
        else:
            raise NotImplementedError()

    def forward(self, input_ids, attention_mask, steer_values):
        self.steer.set_value(steer_values)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids)
        return output

    def parameters(self):
        return self.steer.parameters()

    def state_dict(self):
        return self.steer.state_dict()

    def load_state_dict(self, state_dict):
        self.steer.load_state_dict(state_dict)

    def to_device(self, device):
        self.generator.device = device
        self.model.to(device)
        self.device = device

    def regularization_term(self):
        return self.steer.regularization_term()
