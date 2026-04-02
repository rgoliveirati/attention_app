import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertModel, RobertaTokenizerFast, RobertaModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CARREGAMENTO DE MODELO
# ============================================================
@st.cache_resource
def load_model_and_tokenizer(model_name):
    if "roberta" in model_name.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name, output_attentions=True)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name, output_attentions=True)

    model = model.to(device)
    model.eval()
    return tokenizer, model


# ============================================================
# AGREGAÇÃO DE ATENÇÃO: SUBTOKENS -> PALAVRAS
# ============================================================
def aggregate_attention_to_words(attentions, groups):
    """
    Agrega as matrizes de atenção do nível de subtokens para o nível de palavras.

    attentions: tuple/list de tensores, um por camada, cada um em shape
                [batch, heads, seq_len, seq_len]
    groups: lista de listas com os índices dos subtokens que pertencem a cada palavra.
            Ex.: [[1], [2,3], [4], [5,6,7]]

    Estratégia:
    - Para cada par palavra_i -> palavra_j, calcula a média das atenções entre todos
      os subtokens de i e todos os subtokens de j.
    """
    aggregated_layers = []

    for layer_attn in attentions:
        attn = layer_attn[0].detach().cpu().numpy()  # [heads, seq_len, seq_len]
        n_heads = attn.shape[0]
        n_words = len(groups)

        agg = np.zeros((1, n_heads, n_words, n_words), dtype=np.float32)

        for h in range(n_heads):
            for i, grp_i in enumerate(groups):
                for j, grp_j in enumerate(groups):
                    block = attn[h][np.ix_(grp_i, grp_j)]
                    agg[0, h, i, j] = float(block.mean()) if block.size > 0 else 0.0

        aggregated_layers.append(torch.tensor(agg, dtype=torch.float32))

    return tuple(aggregated_layers)


# ============================================================
# ANÁLISE DA ATENÇÃO
# ============================================================
def analyze_attention(sentence, tokenizer, model, use_offset_mapping=True, show_special_tokens=False):
    tokenizer_kwargs = {
        "text": sentence,
        "return_tensors": "pt",
        "add_special_tokens": True,
    }

    if use_offset_mapping:
        tokenizer_kwargs["return_offsets_mapping"] = True

    inputs = tokenizer(**tokenizer_kwargs)

    offsets = None
    if use_offset_mapping and "offset_mapping" in inputs:
        offsets = inputs["offset_mapping"][0].tolist()

    model_inputs = {k: v.to(device) for k, v in inputs.items() if k != "offset_mapping"}

    input_ids = model_inputs["input_ids"][0].detach().cpu()
    tokenized_text = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        outputs = model(**model_inputs)
        attentions = outputs.attentions

    # Tokens especiais possíveis
    pad_tokens = {"[PAD]", "<pad>"}
    visible_special_tokens = {"[CLS]", "[SEP]", "<s>", "</s>"}

    # ------------------------------------------------------------
    # MODO 1: usa offset_mapping + reconstrói palavras + agrega atenção
    # ------------------------------------------------------------
    if use_offset_mapping:
        filtered_tokens = []
        filtered_offsets = []
        word_groups = []

        current_word_idx = -1

        for idx, token in enumerate(tokenized_text):
            # PAD sempre fora
            if token in pad_tokens:
                continue

            # Tokens especiais opcionais
            if token in visible_special_tokens:
                if show_special_tokens:
                    filtered_tokens.append(token)
                    filtered_offsets.append(offsets[idx] if offsets is not None else None)
                    word_groups.append([idx])
                    current_word_idx += 1
                continue

            # WordPiece do BERT
            if token.startswith("##"):
                piece = token.replace("##", "")
                if (
                    filtered_tokens
                    and filtered_tokens[-1] not in visible_special_tokens
                ):
                    filtered_tokens[-1] += piece
                    word_groups[current_word_idx].append(idx)
                else:
                    # fallback defensivo
                    filtered_tokens.append(piece)
                    filtered_offsets.append(offsets[idx] if offsets is not None else None)
                    word_groups.append([idx])
                    current_word_idx += 1
            else:
                filtered_tokens.append(token)
                filtered_offsets.append(offsets[idx] if offsets is not None else None)
                word_groups.append([idx])
                current_word_idx += 1

        aggregated_attentions = aggregate_attention_to_words(attentions, word_groups)

        return (
            filtered_tokens,
            filtered_offsets if offsets is not None else None,
            aggregated_attentions,
        )

    # ------------------------------------------------------------
    # MODO 2: não usa offset_mapping + mantém subtokens crus
    # ------------------------------------------------------------
    else:
        valid_indices = []

        for i, tok in enumerate(tokenized_text):
            if tok in pad_tokens:
                continue
            if tok in visible_special_tokens and not show_special_tokens:
                continue
            valid_indices.append(i)

        final_tokens = [tokenized_text[i] for i in valid_indices]
        final_offsets = None

        filtered_layers = []
        for layer_attn in attentions:
            attn = layer_attn[0].detach().cpu().numpy()  # [heads, seq_len, seq_len]
            attn = attn[:, valid_indices][:, :, valid_indices]  # [heads, n_valid, n_valid]
            filtered_layers.append(torch.tensor(attn[np.newaxis, ...], dtype=torch.float32))

        return final_tokens, final_offsets, tuple(filtered_layers)


# ============================================================
# PLOT
# ============================================================
def plot_attn(tokens, attns, heads):
    width = 3
    word_height = 1
    pad = 0.1

    cols = []
    count = 10

    for ei, (layer, head) in enumerate(heads):
        count += 1
        if count >= len(cols):
            cols = st.columns(4)
            count = 0

        fig = plt.figure(figsize=(5, 6))
        attn = attns[layer][0, head].detach().cpu().numpy()
        n_words = len(tokens)

        yoffset = 1
        xoffset = 0

        plt.title(f"Layer {layer + 1}, Head {head + 1}")
        plt.axis("off")

        for position, word in enumerate(tokens):
            plt.text(
                xoffset + 0,
                yoffset - position * word_height,
                word,
                ha="right",
                va="center"
            )
            plt.text(
                xoffset + width,
                yoffset - position * word_height,
                word,
                ha="left",
                va="center"
            )

        for i in range(n_words):
            for j in range(n_words):
                plt.plot(
                    [xoffset + pad, xoffset + width - pad],
                    [yoffset - word_height * i, yoffset - word_height * j],
                    color="blue",
                    linewidth=1,
                    alpha=float(attn[i, j]) if i < attn.shape[0] and j < attn.shape[1] else 0.0
                )

        with cols[count]:
            st.pyplot(fig)


# ============================================================
# INTERFACE
# ============================================================
st.title("Análise de Atenção Ampla")

model_options = {
    "BERT Base Uncased": "bert-base-uncased",
    "BERTimbau Base Portuguese Cased": "neuralmind/bert-base-portuguese-cased",
    "mBERT Base Multilingual Uncased": "bert-base-multilingual-uncased",
    "mBERT Base Multilingual Cased": "bert-base-multilingual-cased",
    "RoBERTa Base": "roberta-base",
}

selected_model = st.selectbox("Escolha o modelo:", list(model_options.keys()))
model_name = model_options[selected_model]
tokenizer, model = load_model_and_tokenizer(model_name)

st.caption(f"Dispositivo em uso: {device}")

use_offset_mapping = st.toggle(
    "Usar offset_mapping do tokenizer fast",
    value=True,
    help=(
        "Quando ativado, usa offset_mapping, reconstrói palavras e agrega a atenção "
        "de subtokens para palavras. Quando desativado, mantém subtokens originais."
    )
)

show_special_tokens = st.toggle(
    "Mostrar tokens especiais (CLS/SEP)",
    value=False,
    help=(
        "Exibe tokens especiais do modelo, como [CLS], [SEP], <s> e </s>, "
        "tanto na tabela quanto no gráfico."
    )
)

if use_offset_mapping:
    if show_special_tokens:
        st.write("**Modo atual:** Palavras reconstruídas + atenção agregada + tokens especiais visíveis")
    else:
        st.write("**Modo atual:** Palavras reconstruídas + atenção agregada")
else:
    if show_special_tokens:
        st.write("**Modo atual:** Subtokens originais + tokens especiais visíveis")
    else:
        st.write("**Modo atual:** Subtokens originais")

uploaded_file = st.file_uploader("Carregue o arquivo CSV com sentenças:", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "sentence" in df.columns and "rule" in df.columns:
        sentence_options = df["sentence"].tolist()
        selected_sentence = st.selectbox("Escolha a frase:", sentence_options)
        rule = df[df["sentence"] == selected_sentence]["rule"].values[0]

        st.subheader("Informações da Sentença Selecionada")
        st.write(f"**Regra Gramatical:** {rule}")

        layers = st.slider("Escolha a camada:", 1, 12, 1)
        heads_per_layer = st.slider("Escolha a cabeça:", 1, 12, 1)

        if st.button("Analisar"):
            tokens, offsets, attentions = analyze_attention(
                selected_sentence,
                tokenizer,
                model,
                use_offset_mapping=use_offset_mapping,
                show_special_tokens=show_special_tokens,
            )

            max_layers = len(attentions)
            max_heads = attentions[0].shape[1]

            heads = [
                (layer, head)
                for layer in range(min(layers, max_layers))
                for head in range(min(heads_per_layer, max_heads))
            ]

            st.divider()

            with st.expander("Tokens e offsets", expanded=False):
                df_tokens = pd.DataFrame({
                    "token": tokens,
                    "offset": offsets if offsets is not None else [None] * len(tokens)
                })
                st.dataframe(df_tokens, use_container_width=True)

            with st.expander("Diagnóstico da matriz de atenção", expanded=False):
                st.write(f"Quantidade de itens exibidos: {len(tokens)}")
                st.write(f"Shape da atenção da camada 1: {tuple(attentions[0].shape)}")
                st.write(
                    "A dimensão da matriz de atenção acompanha exatamente os itens exibidos na tabela e no gráfico."
                )

            plot_attn(tokens, attentions, heads)
    else:
        st.error("O arquivo CSV deve conter as colunas 'sentence' e 'rule'.")
else:
    st.warning("Por favor, carregue um arquivo CSV para continuar.")