import torch
import torch.nn as nn
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from .model_utils import Hack_no_grad
from .steers import Projected_Adaptor
from lm_steer.utils import set_seed


class Switching_GPTNeoXModel(nn.Module):
    def __init__(self, model_name, adapted_component, adaptor_class,
                 num_steers, rank, epsilon, init_var,
                 low_resource_mode):
        super().__init__()
        self.adapted_component = adapted_component
        if low_resource_mode:
            self.model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
        else:
            self.model = GPTNeoXForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.init_var = init_var
        self.num_steers = num_steers
        self.device = torch.device("cpu")
        embed_dim = self.model.embed_out.weight.shape[1]
        vocab_size = self.model.embed_out.weight.shape[0]
        self.low_resource_mode = low_resource_mode

        for _param in self.model.parameters():
            _param.requires_grad_(False)

        if adapted_component == "final_layer":
            self.model.gpt_neox = Hack_no_grad(self.model.gpt_neox)
            self.steer = Projected_Adaptor(
                self.model.embed_out, adaptor_class, num_steers, embed_dim,
                vocab_size, rank, epsilon, init_var, "output")
            self.model.set_output_embeddings(self.steer)
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
        self.model.to(device)
        self.device = device

    def regularization_term(self):
        return self.steer.regularization_term()

    def generate(self, prompt, steer_values, min_length=20, max_length=100,
                 seed=None, num_beams=1, num_beam_groups=1, do_sample=True,
                 temperature=1, top_p=1):
        '''
        prompt: a string
        steer_values
        min_length: minimum generation length
        max_length: maximum generation length
        seed: seed for generation. None if not specified.
        '''
        if seed is not None:
            set_seed(seed)
        steer_values = torch.Tensor(steer_values).to(
            self.device)
        if self.low_resource_mode:
            fp16 = torch.float16
            steer_values = steer_values.to(fp16)
            self.steer.projector1.data = self.steer.projector1.to(fp16)
            self.steer.projector2.data = self.steer.projector2.to(fp16)
        self.steer.set_value(steer_values[None])
        with torch.no_grad():
            input_ids = self.tokenizer(
                prompt, return_tensors="pt").input_ids.to(self.device)
            gen_tokens = self.model.generate(
                input_ids,
                num_beams=num_beams, num_beam_groups=num_beam_groups,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
                min_length=min_length, max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id)
            text = self.tokenizer.batch_decode(gen_tokens)[0]

        # recovering
        if self.low_resource_mode:
            fp32 = torch.float32
            self.steer.projector1.data = self.steer.projector1.to(fp32)
            self.steer.projector2.data = self.steer.projector2.to(fp32)
        return text
