# https://huggingface.co/spaces/Glaciohound/LM-Steer

import torch
import streamlit as st
import random
import numpy as np
import pandas as pd
from lm_steer.models.get_model import get_model


@st.cache_resource(show_spinner="Loading model...")
def st_get_model(model_name, low_resource_mode):
    device = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device("cpu")
    model, tokenizer = get_model(
        model_name, "final_layer", "multiply",
        4,
        1000, 1e-3, 1e-2, low_resource_mode
    )
    model.to_device(device)
    ckpt = torch.load(f"checkpoints/{model_name}.pt", map_location=device)
    model.load_state_dict(ckpt[1])
    return model, tokenizer


def word_embedding_space_analysis(model, tokenizer, dim):
    matrix = model.steer.projector1.data[dim].matmul(
        model.steer.projector2.data[dim].transpose(0, 1))
    S, V, D = torch.linalg.svd(matrix)
    embeddings = model.steer.lm_head.weight

    data = []
    for _i in range(10):
        left_tokens = embeddings.matmul(D[_i]).argsort()[-20:].flip(0)
        right_tokens = embeddings.matmul(D[_i]).argsort()[:20]

        def filter_words(side_tokens):
            output = []
            for t in side_tokens:
                word = tokenizer.decode([t])
                if not word[0].isalpha() and word[1:].isalpha():
                    output.append(word[1:]+"-")
            return output

        data.append([
            ", ".join(filter_words(side_tokens))
            for side_tokens in [left_tokens, right_tokens]
        ])
    st.table(pd.DataFrame(
        data,
        columns=["One Direction", "Another Direction"],
        index=[f"Dim {_i}" for _i in range(10)],
    ))


def main():
    # set up the page
    random.seed(0)
    title = "LM-Steer: Word Embeddings Are Steers for Language Models"
    st.set_page_config(
        layout="wide",
        page_title=title,
        page_icon="🛞",
    )
    st.title(title)
    '''
    Live demo for the paper ["**LM-Steer: Word Embeddings Are Steers for
    Language Models**"](https://arxiv.org/abs/2305.12798) (**ACL 2024
    Outstanding Paper Award**) by Chi Han, Jialiang Xu, Manling Li, Yi Fung,
    Chenkai Sun, Nan Jiang, Tarek Abdelzaher, Heng Ji. GitHub repository:
    https://github.com/Glaciohound/LM-Steer.
    '''
    st.subheader("Overview")
    st.image('https://raw.githubusercontent.com/Glaciohound/LM-Steer'
             '/refs/heads/main/assets/overview_fig.jpg')
    '''
    Language models (LMs) automatically learn word embeddings during
    pre-training on language corpora. Although word embeddings are usually
    interpreted as feature vectors for individual words, their roles in
    language model generation remain underexplored. In this work, we
    theoretically and empirically revisit output word embeddings and find that
    their linear transformations are equivalent to steering language model
    generation styles. We name such steers LM-Steers and find them existing in
    LMs of all sizes. It requires learning parameters equal to 0.2% of the
    original LMs' size for steering each style.
    '''

    # set up the model
    st.divider()
    st.divider()
    st.subheader("Select a model:")
    '''
    Due to resource limits, we are only able to provide a few models for
    steering. You can also refer to the Github repository:
    https://github.com/Glaciohound/LM-Steer for hosting larger models.
    '''
    col1, col2 = st.columns(2)
    st.session_state.model_name = col1.selectbox(
        "Select a model to steer",
        [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            # "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b",
            # "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b",
            # "EleutherAI/gpt-j-6B",
        ],
    )
    low_resource_mode = True if st.session_state.model_name in (
        "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b", "EleutherAI/gpt-j-6B",
    ) else False
    model, tokenizer = st_get_model(
        st.session_state.model_name, low_resource_mode)
    num_param = model.steer.projector1.data.shape[1] ** 2 / 1024 ** 2
    total_param = sum(p.numel() for _, p in model.named_parameters()) / \
        1024 ** 2
    ratio = num_param / total_param
    col2.write(f"Steered {num_param:.1f}M out of {total_param:.1f}M "
               "parameters, ratio: {:.2%}".format(ratio))

    # steering
    steer_range = 4.
    steer_interval = 0.5
    st.subheader("Enter a sentence and steer the model")
    st.session_state.prompt = st.text_input(
        "Enter a prompt",
        st.session_state.get("prompt", "My life")
    )
    # col1, col2, col3 = st.columns(3, gap="medium")
    col1, col2, col3 = st.columns([2, 2, 1], gap="medium")
    sentiment = col1.slider(
        "Sentiment (the larger the more positive)",
        -steer_range, steer_range, 3.0, steer_interval)
    detoxification = col2.slider(
        "Detoxification Strength (the larger the less toxic)",
        -steer_range, steer_range, 0.0,
        steer_interval)
    max_length = col3.number_input("Max length", 50, 300, 50, 50)
    col1, col2, col3, _ = st.columns(4)
    randomness = col2.checkbox("Random sampling", value=False)

    if "output" not in st.session_state:
        st.session_state.output = ""
    if col1.button("Steer and generate!", type="primary"):
        with st.spinner("Generating..."):
            steer_values = [detoxification, 0, sentiment, 0]
            st.session_state.output = model.generate(
                st.session_state.prompt,
                steer_values,
                seed=None if randomness else 0,
                min_length=0,
                max_length=max_length,
                do_sample=True,
            )
    analyzed_text = \
        st.text_area("Generated text:", st.session_state.output, height=200)

    # Analysing the sentence
    st.divider()
    st.divider()
    st.subheader("Analyzing Styled Texts")
    '''
    LM-Steer also serves as a probe for analyzing the text. It can be used to
    analyze the sentiment and detoxification of the text. Now, we proceed and
    use LM-Steer to analyze the text in the box above. You can also modify the
    text or use your own. Please note that these two dimensions can be
    entangled, as a negative sentiment may also detoxify the text.
    '''
    if st.session_state.get("output", "") != "" and \
            st.button("Analyze the styled text", type="primary"):
        col1, col2 = st.columns(2)
        for name, col, dim, color in zip(
            ["Sentiment", "Detoxification"],
            [col1, col2],
            [2, 0],
            ["#ff7f0e", "#1f77b4"],
        ):
            with st.spinner(f"Analyzing {name}..."):
                col.subheader(name)
                # classification
                col.markdown(
                    "##### Dimension-Wise Classification Distribution")
                _, dist_list, _ = model.steer_analysis(
                    analyzed_text,
                    dim, -steer_range, steer_range,
                    bins=2*int(steer_range)+1,
                )
                dist_list = np.array(dist_list)
                col.bar_chart(
                    pd.DataFrame(
                        {
                            "Value": dist_list[:, 0],
                            "Probability": dist_list[:, 1],
                        }
                    ), x="Value", y="Probability",
                    color=color,
                )

                # key tokens
                pos_steer, neg_steer = np.zeros((2, 4))
                pos_steer[dim] = 1
                neg_steer[dim] = -1
                _, token_evidence = model.evidence_words(
                    analyzed_text,
                    [pos_steer, neg_steer],
                )
                tokens = tokenizer(analyzed_text).input_ids
                tokens = [f"{i:3d}: {tokenizer.decode([t])}"
                          for i, t in enumerate(tokens)]
                col.markdown("##### Token's Evidence Score in the Dimension")
                col.write("The polarity of the token's evidence score "
                          "which aligns with sliding bar directions."
                          )
                col.bar_chart(
                    pd.DataFrame(
                        {
                            "Token": tokens[1:],
                            "Evidence": token_evidence,
                        }
                    ), x="Token", y="Evidence",
                    horizontal=True, color=color,
                )

    st.divider()
    st.divider()
    st.subheader("The Word Embeddings Space Analysis")
    '''
    LM-Steer provides a lens on how word embeddings correlate with LM word
    embeddings: what word dimensions contribute to or contrast to a specific
    style. This analysis can be used to understand the word embedding space
    and how it steers the model's generation.
    Note that due to the bidirectional nature of the embedding spaces, in each
    dimension, sometimes only one side of the word embeddings is most relevant
    to the style (can be either left or right).
    '''
    dimension = st.selectbox(
        "Select a dimension to analyze",
        ["Sentiment", "Detoxification"],
    )
    dim = 2 if dimension == "Sentiment" else 0
    with st.spinner("Analyzing..."):
        word_embedding_space_analysis(model, tokenizer, dim)


if __name__ == "__main__":
    main()
