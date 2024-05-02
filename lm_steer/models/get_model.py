
def get_model(model_name, adapted_component, adaptor_class, num_steers, rank,
              epsilon, init_var, low_resource_mode):
    if model_name.startswith("EleutherAI/gpt-neo") or \
            model_name.startswith("gpt2"):
        from lm_steer.models.model_gpt_neo import Switching_GPTNeoModel
        model = Switching_GPTNeoModel(
            model_name, adapted_component, adaptor_class, num_steers, rank,
            epsilon, init_var, low_resource_mode)
        return model, model.tokenizer
    elif model_name.startswith("lora-gpt2"):
        from lm_steer.models.model_lora_gpt_neo import LORA_GPTNeoModel
        model = LORA_GPTNeoModel(model_name, rank, epsilon)
        return model, model.tokenizer
    elif model_name.startswith("embedding_tuning"):
        from lm_steer.models.model_embedding_tuning_gpt_neo import \
            EmbeddingTuning_GPTNeoModel
        model = EmbeddingTuning_GPTNeoModel(model_name)
        return model, model.tokenizer
    elif model_name.startswith("prefix-gpt2"):
        from lm_steer.models.model_prefix_gpt_neo import PREFIX_GPTNeoModel
        model = PREFIX_GPTNeoModel(model_name)
        return model, model.tokenizer
    elif model_name.startswith("EleutherAI/pythia"):
        from lm_steer.models.model_gpt_neox import Switching_GPTNeoXModel
        model = Switching_GPTNeoXModel(
            model_name, adapted_component, adaptor_class, num_steers, rank,
            epsilon, init_var, low_resource_mode)
        return model, model.tokenizer
    elif model_name.startswith("EleutherAI/gpt-j"):
        from lm_steer.models.model_gpt_j import Switching_GPTJModel
        model = Switching_GPTJModel(
            model_name, adapted_component, adaptor_class, num_steers, rank,
            epsilon, init_var, low_resource_mode)
        return model, model.tokenizer
    elif model_name.startswith("microsoft/DialoGPT"):
        from lm_steer.models.model_dialogpt import Switching_DialoGPTModel
        model = Switching_DialoGPTModel(
            model_name, adapted_component, adaptor_class, num_steers, rank,
            epsilon, init_var, low_resource_mode)
        return model, model.tokenizer
    else:
        raise NotImplementedError()
