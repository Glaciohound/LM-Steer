import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from IPython import embed

from lm_steer.arguments import parse_args
from lm_steer.models.get_model import get_model
from lm_steer.utils import RunningMean

from data import load_dataset


def args_assert(args):
    assert args.prompt_only or args.steer_values is not None or \
        args.model_name.startswith("lora-") or \
        args.model_name.startswith("embedding_tuning")


def generate(prompt_data, steer_values, tokenizer, model,
             prompt_num, prompt_length, num_beams, num_beam_groups,
             do_sample, temperature, top_p, device):
    for _prompt in tqdm(prompt_data):
        _prompt["generations"] = []
        prompt_text = _prompt["prompt"]["text"]
        token_length = tokenizer(prompt_text,
                                 return_tensors="pt")["input_ids"].shape[1]
        for _i in range(prompt_num):
            output = model.generate(
                prompt_text,
                steer_values,
                seed=_i,
                max_length=token_length+prompt_length,
                min_length=token_length+prompt_length,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            output = output[len(prompt_text):]
            _prompt["generations"].append({
                "text": output
            })
        if args.verbose:
            print(prompt_text)
            print(_prompt["generations"])


def main(args):
    args_assert(args)
    train_data = load_dataset(args.dataset_name, args.data_dir, args.subset)
    dataloader = DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True)
    data_iter = iter(dataloader)

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, args.adaptor_class,
        args.num_steers,
        args.rank, args.epsilon, args.init_var, args.low_resource_mode)
    model.to_device(device)

    start_step = 0
    if os.path.exists(args.ckpt_name) or args.eval_only:
        ckpt = torch.load(args.ckpt_name)
        model.load_state_dict(ckpt[1])
        start_step = ckpt[2]
        print(f"resume training from {start_step}")
    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    if args.eval_only:
        args.n_steps = 0

    pbar = tqdm(range(start_step, args.n_steps))
    loss_mean = RunningMean(args.gamma_mean)
    scaler = torch.cuda.amp.GradScaler()

    for step_i in pbar:
        batch = next(data_iter, None)
        if batch is None:
            data_iter = iter(dataloader)
            batch = next(data_iter, None)

        batch_text = batch["text"]
        batch_stance = torch.Tensor(batch["label"])[:, None].to(device)
        if args.pos_neg_independent:
            batch_stance = torch.cat([
                (batch_stance > 0).float(),
                (batch_stance < 0).float(),
            ], 1)
        if args.dummy_steer:
            batch_stance = torch.cat(
                [batch_stance, torch.ones_like(batch_stance[:, 0])[:, None]],
                1)
        tokenized = tokenizer(batch_text, padding=True,
                              max_length=args.max_length, truncation=True)
        input_ids = torch.LongTensor(tokenized["input_ids"]).to(device)

        optimizer.zero_grad()
        attention_mask = torch.LongTensor(tokenized["attention_mask"]).to(
            device)
        if args.low_resource_mode:
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                loss = model(
                    input_ids, attention_mask,
                    batch_stance.float()
                ).loss
                regularization_term = model.regularization_term()
            scaler.scale(loss + args.regularization * regularization_term
                         ).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(
                input_ids, attention_mask,
                batch_stance.float()
            ).loss
            regularization_term = model.regularization_term()
            (loss + args.regularization * regularization_term).backward()
            optimizer.step()

        loss_mean.update(loss)
        pbar.set_description(
            f"{loss_mean.value}, {regularization_term.item()}")
        if (step_i+1) % args.log_step == 0:
            print(pbar.desc, flush=True)

    if not args.eval_only and args.ckpt_name is not None:
        torch.save([args, model.state_dict(), args.n_steps], args.ckpt_name)

    if args.prompt_only:
        embed()
        exit()

    # predicting sentences
    with open(args.eval_file, "r") as f:
        prompt_data = list(map(json.loads, f.readlines()))

    model.eval()
    prompt_num = 25
    prompt_length = 20
    if args.eval_size is not None:
        prompt_data = prompt_data[:args.eval_size]
    num_beams = 1
    num_beam_groups = 1
    do_sample = True
    temperature = args.temperature
    steer_values = list(map(float, args.steer_values)) \
        if args.steer_values is not None else None

    generate(prompt_data, steer_values, tokenizer, model, prompt_num,
             prompt_length, num_beams, num_beam_groups, do_sample, temperature,
             args.top_p, device)

    with open(args.output_file, "w") as f:
        for _prompt in prompt_data:
            f.write(json.dumps(_prompt) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
