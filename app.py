# https://huggingface.co/spaces/Glaciohound/LM-Steer

import torch
import nltk
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


@st.cache_data()
def word_embedding_space_analysis(
        model_name, dim):
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    projector1 = model.steer.projector1.data[dim]
    projector2 = model.steer.projector2.data[dim]
    embeddings = model.steer.lm_head.weight
    matrix = projector1.matmul(projector2.transpose(0, 1))
    S, V, D = torch.linalg.svd(matrix)

    data = []
    top = 50
    select_words = 20
    n_dim = 10
    for _i in range(n_dim):
        left_tokens = embeddings.float().matmul(D[_i]).argsort()[-top:].flip(0)
        right_tokens = embeddings.float().matmul(D[_i]).argsort()[:top]

        def filter_words(side_tokens):
            output = []
            for t in side_tokens:
                word = tokenizer.decode([t])
                if (
                    len(word) > 2 and word[0] == " " and
                    word[1:].isalpha() and word[1:].lower().islower()
                ):
                    word = word[1:]
                    if word.lower() in nltk.corpus.words.words():
                        output.append(word)
            return output

        left_tokens = filter_words(left_tokens)
        right_tokens = filter_words(right_tokens)
        if len(left_tokens) < len(right_tokens):
            left_tokens = right_tokens
        data.append(", ".join(left_tokens[:select_words]))
    return pd.DataFrame(
        data,
        columns=["Words Contributing to the Style"],
        index=[f"Dim#{_i}" for _i in range(n_dim)],
    ), D


# rgb tuple to hex color
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def main():
    # set up the page
    random.seed(0)
    nltk.download('words')
    dimension_names = ["Sentiment", "Detoxification"]
    dimension_colors = ["#ff7f0e", "#1f77b4"]
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
    col1, col2, col3 = st.columns([1, 5, 1])
    col2.image(
        'https://raw.githubusercontent.com/Glaciohound/LM-Steer'
        '/refs/heads/main/assets/overview_fig.jpg',
        caption="LM-Steer Method Overview"
    )
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
    st.subheader("Select A Model and Steer It")
    '''
    Due to resource limits, we are only able to provide a few models for
    steering. You can also refer to the Github repository:
    https://github.com/Glaciohound/LM-Steer to host larger models locally.
    Some generated texts may contain toxic or offensive content. Please be
    cautious when using the generated texts.
    Note that for these smaller models, the generation quality may not be as
    good as the larger models (GPT-4, Llama, etc.).
    '''
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    model_name = col1.selectbox(
        "Select a model to steer",
        [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
            "EleutherAI/gpt-j-6B",
        ],
    )
    low_resource_mode = True if model_name in (
        "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b", "EleutherAI/gpt-j-6B",
    ) and torch.cuda.is_available() else False
    # low_resource_mode = False
    model, tokenizer = st_get_model(
        model_name, low_resource_mode)
    st.session_state.model = model
    st.session_state.tokenizer = tokenizer
    num_param = model.steer.projector1.data.shape[1] ** 2 / 1024 ** 2
    total_param = sum(p.numel() for _, p in model.named_parameters()) / \
        1024 ** 2
    ratio = num_param / total_param
    col2.metric("Parameters Steered", f"{num_param:.1f}M")
    col3.metric("LM Total Size", f"{total_param:.1f}M")
    col4.metric("Steered Ratio", f"{ratio:.2%}")

    # steering
    steer_range = 3.
    steer_interval = 0.2
    st.session_state.prompt = st.text_input(
        "Enter a prompt",
        st.session_state.get("prompt", "My life")
    )
    col1, col2, col3 = st.columns([2, 2, 1], gap="medium")
    sentiment = col1.slider(
        "Sentiment (Negative ↔︎ Positive)",
        -steer_range, steer_range, 0.0, steer_interval)
    detoxification = col2.slider(
        "Detoxification Strength (Toxic ↔︎ Clean)",
        -steer_range, steer_range, 0.0,
        steer_interval)
    max_length = col3.number_input("Max length", 20, 300, 20, 40)
    col1, col2, col3, _ = st.columns(4)
    randomness = col2.checkbox("Random sampling", value=False)

    if "output" not in st.session_state:
        st.session_state.output = ""
    if col1.button("Steer and generate!", type="primary"):
        if sentiment == 0 and detoxification == 0:
            '''
            **The steer values are both 0, which means the steered model
            is the same as the original model.**
            '''
        with st.spinner("Generating..."):
            steer_values = [detoxification, 0, sentiment, 0]
            st.session_state.output = model.generate(
                st.session_state.prompt,
                steer_values,
                seed=None if randomness else 0,
                min_length=0,
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
            )

    with st.chat_message("human"):
        st.write(st.session_state.output)

    # Analysing the sentence
    st.divider()
    st.divider()
    st.subheader("LM-Steer Converts Any LM Into A Text Analyzer")
    '''
    LM-Steer also serves as a probe for analyzing the text. It can be used to
    analyze the sentiment and detoxification of the text. Now, we proceed and
    use LM-Steer to analyze the text in the box above. You can also modify the
    text or use your own. You may observe that these two dimensions can be
    entangled, as a negative sentiment may also detoxify the text.
    '''
    st.session_state.analyzed_text = \
        st.text_area("Text to analyze:", st.session_state.output, height=200)
    if st.session_state.get("analyzed_text", "") != "" and \
            st.button("Analyze the text above", type="primary"):
        col1, col2 = st.columns(2)
        for name, col, dim, color, axis_annotation in zip(
            dimension_names,
            [col1, col2],
            [2, 0],
            dimension_colors,
            ["Negative ↔︎ Positive", "Toxic ↔︎ Clean"]
        ):
            with st.spinner(f"Analyzing {name}..."):
                col.subheader(name)
                # classification
                col.markdown(
                    "##### Sentence Classification Distribution")
                col.write(axis_annotation)
                _, dist_list, _ = model.steer_analysis(
                    st.session_state.analyzed_text,
                    dim, -steer_range, steer_range,
                    bins=4*int(steer_range)+1,
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
                    st.session_state.analyzed_text,
                    [pos_steer, neg_steer],
                )
                tokens = tokenizer(st.session_state.analyzed_text).input_ids
                tokens = [f"{i:3d}: {tokenizer.decode([t])}"
                          for i, t in enumerate(tokens)]
                col.markdown("##### Token's Evidence Score in the Dimension")
                col.write(axis_annotation)
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
    st.subheader("LM-Steer Unveils Word Embeddings Space")
    '''
    LM-Steer provides a lens on how word embeddings correlate with LM word
    embeddings: what word dimensions contribute to or contrast to a specific
    style. This analysis can be used to understand the word embedding space
    and how it steers the model's generation.
    '''
    for dimension, color in zip(dimension_names, dimension_colors):
        f'##### {dimension} Word Dimensions'
        dim = 2 if dimension == "Sentiment" else 0
        analysis_result, D = word_embedding_space_analysis(
            model_name, dim)
        with st.expander("Show the analysis results"):
            color_scale = 7
            color_init = 230
            st.table(analysis_result.style.apply(
                lambda x: [
                    "background: " + rgb_to_hex(
                        (255,
                         color_init-(9-i)*color_scale,
                         color_init-(9-i)*color_scale)
                        if dimension == "Sentiment" else
                        (color_init-(9-i)*color_scale,
                         color_init-(9-i)*color_scale,
                         255)
                    )
                    for i in range(len(x))
                ]
            ))
            embeddings = model.steer.lm_head.weight
            dim1 = embeddings.float().matmul(D[0]).tolist()
            dim2 = embeddings.float().matmul(D[1]).tolist()
            words = [tokenizer.decode([i]) for i in range(len(embeddings))]
            scatter_chart = [
                (_d1, _d2, _word)
                for _d1, _d2, _word in zip(dim1, dim2, words)
                if len(_word) > 2 and _word[0] == " " and
                _word[1:].isalpha() and _word[1:].lower().islower()
            ]
            scatter_chart = pd.DataFrame(
                scatter_chart,
                columns=["Dim1", "Dim2", "Word"]
            )
            st.scatter_chart(
                scatter_chart, x="Dim1", y="Dim2",
                color="Word",
                # color=color,
                height=1000, size=50,)


if __name__ == "__main__":
    main()
