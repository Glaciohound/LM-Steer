import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline

from .model_utils import Hack_no_grad, find_max_subspans
from .steers import Projected_Adaptor
from lm_steer.utils import set_seed


punctuations = [
    '!', '"', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
    # '/', '#',
    ':', ';', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`',
    '{', '|', '}', '~',
    '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', '·',
    '¸', '¹', 'º', '»', '¼', '½', '¾',
    '\n', ' ',
]


class Switching_GPTNeoModel(nn.Module):
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
        self.steer.set_value(steer_values[None])
        with torch.no_grad():
            text = self.generator(
                prompt, num_beams=num_beams, num_beam_groups=num_beam_groups,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
                min_length=min_length, max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text = text[0]["generated_text"]
        return text

    def evidence_words(self, prompt, comparing_steer_values,
                       truncation_length=1024, max_segments=4, max_length=10):
        if isinstance(comparing_steer_values, list):
            comparing_steer_values = \
                torch.Tensor(comparing_steer_values).to(self.device)
        if (comparing_steer_values[0] - comparing_steer_values[1]
                ).abs().sum() <= 0.2:
            return [(prompt, None)]
        tokenized = self.tokenizer(
            prompt, return_tensors="pt", max_length=truncation_length, truncation=True)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(self.device)
        input_ids = input_ids.expand(2, -1)
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(
            self.device)
        attention_mask = attention_mask.expand(2, -1)
        self.steer.set_value(comparing_steer_values)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids)
        length = input_ids.shape[1]
        loss_token = F.cross_entropy(
            output.logits[:, :-1].reshape((2)*(length-1), -1),
            input_ids[:, 1:].reshape(-1),
            reduction="none"
        )
        loss_token = loss_token.reshape(2, length - 1)

        token_evidence = (- loss_token[0] + loss_token[1])
        tokens = input_ids[0]
        evidence_segments = find_max_subspans(
            token_evidence.cpu().numpy().tolist(), max_segments, max_length)[0]
        evidence_segments = [
            (_seg[0]+1, _seg[1]+1) for _seg in evidence_segments]
        start = 0
        output = []
        if len(evidence_segments) > 0:
            for _segment in evidence_segments:
                if _segment[0] > start:
                    output.append((
                        self.tokenizer.decode(tokens[start: _segment[0]]),
                        None
                    ))
                output.append((
                    self.tokenizer.decode(tokens[_segment[0]: _segment[1]]),
                    "evidence"
                ))
                start = _segment[1]
            length = tokens.shape[-1]
            if _segment[1] < length:
                output.append((
                    self.tokenizer.decode(tokens[_segment[1]: length]),
                    None
                ))
        else:
            output = [(prompt, None)]

        return output

    def steer_analysis(self, prompt, steer_dim, min_value=-3, max_value=3,
                        bins=7):
        tokenized = self.tokenizer(prompt)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(self.device)
        input_ids = input_ids.expand(bins + 1, -1)
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(
            self.device)
        attention_mask = attention_mask.expand(bins + 1, -1)
        steer_values = torch.zeros(bins+1, self.num_steers).to(self.device)
        for bin_i in range(bins):
            steer_values[bin_i, steer_dim] = (
                min_value + (max_value - min_value) / (bins - 1) * bin_i
            )
        self.steer.set_value(steer_values)
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids)
        length = input_ids.shape[1]
        loss_token = F.cross_entropy(
            output.logits[:, :-1].reshape((bins+1)*(length-1), -1),
            input_ids[:, 1:].reshape(-1),
            reduction="none"
        )
        loss_token = loss_token.reshape(bins + 1, length - 1)
        loss = loss_token.mean(-1)[:-1]
        dist = ((- loss + loss.mean()) * 100).softmax(0)
        dist_list = list(zip(
            [
                min_value + (max_value - min_value) / (bins - 1) * bin_i
                for bin_i in range(bins)
            ],
            dist.tolist(),
        ))
        best_guess = loss.argmin(0)
        best_guess_value = min_value + \
            (max_value - min_value) / (bins - 1) * best_guess.item()

        token_evidence = (- loss_token[best_guess] + loss_token[-1]) * 10
        token_evidence = [0] + token_evidence.tolist()
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        word_evidence_list = []
        start = 0
        n_tokens = len(input_ids[0])
        for token_i in range(1, n_tokens+1):
            span = self.tokenizer.decode(input_ids[0][start: token_i])
            for _punc in punctuations:
                if token_i == n_tokens or _punc in span:
                    new_span = self.tokenizer.decode(
                        input_ids[0][start: token_i-1]).strip()
                    if len(new_span) <= 1:
                        break
                    word_evidence_list.append((
                        new_span,
                        np.array(token_evidence[start: token_i-1]).mean()
                    ))
                    start = token_i - 1
                    break

        # token_evidence_list = list(zip(tokens, token_evidence))
        return best_guess_value, dist_list, word_evidence_list
