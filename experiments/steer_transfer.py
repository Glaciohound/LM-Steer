from tqdm import tqdm
import torch
from lm_steer.arguments import parse_args
from lm_steer.models.get_model import get_model
from torch.optim import Adam


def main(args):
    ckpt = torch.load(args.ckpt_name)
    device = torch.device("cuda:0")

    print("loading model 1", args.model_name)
    to_model, to_tokenizer = get_model(
        args.model_name, args.adapted_component, args.adaptor_class,
        args.num_steers, args.rank, args.epsilon, args.init_var,
        args.low_resource_mode
    )
    print("loading model 2", args.transfer_from)
    from_model, from_tokenizer = get_model(
        args.transfer_from, args.adapted_component, args.adaptor_class,
        args.num_steers, args.rank, args.epsilon, args.init_var,
        args.low_resource_mode
    )

    print("starting to transfer")
    to_embeddings = to_model.steer.lm_head.weight
    from_embeddings = from_model.steer.lm_head.weight
    to_vocab = to_tokenizer.vocab
    from_vocab = from_tokenizer.vocab

    from_vocab_set = set(from_tokenizer.vocab.keys())
    shared_vocab = [
        _v for _v in tqdm(to_tokenizer.vocab.keys())
        if _v in from_vocab_set
    ]

    print("from_embeddings shape:", from_embeddings.shape)
    print("to_embeddings shape:", to_embeddings.shape)
    print("from_vcab size:", len(from_vocab))
    print("to_vocab size:", len(to_vocab))
    print("shared vocab size:", len(shared_vocab))

    from_indices = [from_vocab[_v] for _v in shared_vocab]
    to_indices = [to_vocab[_v] for _v in shared_vocab]

    from_shared_embeddings = from_embeddings[from_indices].to(device).float()
    to_shared_embeddings = to_embeddings[to_indices].to(device).float()

    # B_forward2 = torch.linalg.pinv(from_shared_embeddings).matmul(
    #     to_shared_embeddings).to(device)
    # B_backward2 = torch.linalg.pinv(to_shared_embeddings).matmul(
    #     from_shared_embeddings).to(device)

    B_forward = torch.randn((from_embeddings.shape[1], to_embeddings.shape[1]),
                            requires_grad=True, device=device)
    B_backward = torch.randn((to_embeddings.shape[1],
                              from_embeddings.shape[1]),
                             requires_grad=True, device=device)
    optimizer = Adam([B_forward, B_backward], lr=args.lr)
    top_k = 4000

    pbar = tqdm(range(args.n_steps))
    for step_i in pbar:
        optimizer.zero_grad()
        loss1 = (from_shared_embeddings[:top_k].matmul(B_forward)
                 - to_shared_embeddings[:top_k]).pow(2).mean()
        loss2 = (to_shared_embeddings[:top_k].matmul(B_backward)
                 - from_shared_embeddings[:top_k]).pow(2).mean()
        # loss1 = -F.cosine_similarity(
        #     from_shared_embeddings[:top_k].matmul(B_forward),
        #     to_shared_embeddings[:top_k]
        # ) .mean()
        # loss2 = -F.cosine_similarity(
        #     to_shared_embeddings[:top_k].matmul(B_backward),
        #     from_shared_embeddings[:top_k]
        # ) .mean()
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        pbar.set_description(f"{loss.item()}")

    # from IPython import embed; embed(); exit()

    B_backward = B_backward / B_backward.norm(dim=1)[:, None]
    projector1 = B_backward.matmul(ckpt[1]["projector1"])
    projector2 = B_backward.matmul(ckpt[1]["projector2"])

    save_ckpt = {
        "projector1": projector1,
        "projector2": projector2
    }

    torch.save([None, save_ckpt, ckpt[2]], args.output_file)
    print("written to:", args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
