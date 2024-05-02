import torch
import torch.nn as nn


class Projected_Adaptor(nn.Module):
    def __init__(self, lm_head, adaptor_class, num_steers, embed_dim,
                 vocab_size, rank, epsilon, init_var, position="output"):
        super().__init__()
        assert rank > 0
        if adaptor_class == "multiply":
            self.projector1 = nn.Parameter(torch.randn(
                num_steers, embed_dim, rank
            ) * init_var)
            self.projector2 = nn.Parameter(torch.randn(
                num_steers, embed_dim, rank
            ) * init_var)
        elif adaptor_class == "add":
            self.add_vec = nn.Parameter(torch.randn(
                num_steers, embed_dim
            ))
        elif adaptor_class == "offset":
            self.offset_vec = nn.Parameter(torch.randn(
                num_steers, vocab_size
            ))
        else:
            raise NotImplementedError()

        self.adaptor_class = adaptor_class
        self.rank = rank
        self.lm_head = lm_head
        self.epsilon = epsilon
        self.position = position
        self.num_steers = num_steers
        self.init_var = init_var
        self.steer_values = torch.zeros(num_steers)

    def set_value(self, steer_values):
        self.steer_values = steer_values

    def forward(self, state):
        if self.steer_values.abs().sum() == 0:
            return state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
        if self.adaptor_class == "multiply":
            delta = state[:, None].matmul(self.projector1[None]) *\
                self.steer_values[:, :, None, None]
            delta = delta.matmul(
                self.projector2.transpose(1, 2)[None]).sum(1)
            projected_state = state + self.epsilon * delta
            logits = projected_state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
        elif self.adaptor_class == "add":
            add_values = self.steer_values.matmul(self.add_vec)
            projected_state = state + self.epsilon * add_values[:, None]
            logits = projected_state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
        elif self.adaptor_class == "offset":
            offset_values = self.steer_values.matmul(self.offset_vec)
            logits = state.matmul(
                self.lm_head.weight.detach().transpose(0, 1))
            logits = logits + self.epsilon * offset_values[:, None]
        return logits

    def regularization_term(self):
        if self.adaptor_class == "multiply":
            return self.projector1.pow(2).sum() + self.projector2.pow(2).sum()
        elif self.adaptor_class == "add":
            return self.add_vec.pow(2).sum()
        elif self.adaptor_class == "offset":
            return self.offset_vec.pow(2).sum()

    def parameters(self):
        if self.adaptor_class == "multiply":
            return [self.projector1, self.projector2]
        elif self.adaptor_class == "add":
            return [self.add_vec]
        elif self.adaptor_class == "offset":
            return [self.offset_vec]

    def state_dict(self):
        if self.adaptor_class == "multiply":
            return {"projector1": self.projector1,
                    "projector2": self.projector2}
        elif self.adaptor_class == "add":
            return {"add_vec": self.add_vec}
        elif self.adaptor_class == "offset":
            return {"offset_vec": self.offset_vec}

    def load_state_dict(self, state_dict):
        if self.adaptor_class == "multiply":
            self.projector1.data = state_dict["projector1"]
            self.projector2.data = state_dict["projector2"]
        elif self.adaptor_class == "add":
            self.add_vec.data = state_dict["add_vec"]
        elif self.adaptor_class == "offset":
            self.offset_vec.data = state_dict["offset_vec"]
