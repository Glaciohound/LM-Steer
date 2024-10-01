import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

from .model_utils import Hack_no_grad
from .steers import Projected_Adaptor
from .model_base import LMSteerBase


class Switching_GPTNeoXModel(LMSteerBase):
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
        return super().generate_low_resource(
            prompt, steer_values, min_length, max_length, seed,
            num_beams, num_beam_groups, do_sample, temperature, top_p)
