
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from lm_steer.utils import set_seed
from .model_utils import find_max_subspans


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


class LMSteerBase(nn.Module):
    def evidence_words(self, prompt, comparing_steer_values,
                       truncation_length=1024, max_segments=4, max_length=10):
        if isinstance(comparing_steer_values, list):
            comparing_steer_values = \
                torch.Tensor(comparing_steer_values).to(self.device)
        if (comparing_steer_values[0] - comparing_steer_values[1]).abs().sum()\
                <= 0.2:
            return [(prompt, None)]
        tokenized = self.tokenizer(
            prompt, return_tensors="pt",
            max_length=truncation_length, truncation=True)
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

        return output, token_evidence.tolist()

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
        dist = ((- loss + loss.mean()) * 10).softmax(0)
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
            inputs = self.tokenizer(
                prompt, return_tensors="pt").to(self.device)
            text = self.model.generate(
                **inputs,
                num_beams=num_beams, num_beam_groups=num_beam_groups,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
                min_length=min_length, max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text = self.tokenizer.decode(text[0], skip_special_tokens=True)

        return text

    def generate_low_resource(
        self, prompt, steer_values, min_length=20, max_length=100,
        seed=None, num_beams=1, num_beam_groups=1, do_sample=True,
        temperature=1, top_p=1
    ):
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
        fp32 = torch.float32
        self.steer.projector1.data = self.steer.projector1.to(fp32)
        self.steer.projector2.data = self.steer.projector2.to(fp32)
        return text

    def state_dict(self):
        return self.steer.state_dict()

    def load_state_dict(self, state_dict):
        self.steer.load_state_dict(state_dict)

    def parameters(self):
        return self.steer.parameters()

    def to_device(self, device):
        self.model.to(device)
        self.device = device

    def regularization_term(self):
        return self.steer.regularization_term()

    def forward(self, input_ids, attention_mask, steer_values):
        self.steer.set_value(steer_values)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids)
        return output
