# -*- coding: utf-8 -*-
# ============================================================
# Análise de Atenção — Seleção de Sentença, Padrão e Tokens
# ============================================================

import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertModel
import math

# ------------------------------------------------------------
# Detecta dispositivo: GPU se disponível, caso contrário CPU
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.text(f"Dispositivo: {device}")

# ------------------------------------------------------------
# Carrega modelo e tokenizador (cacheado — não recarrega a cada rerun)
# ------------------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model = model.to(device)
    model.eval()
    return tokenizer, model

# ------------------------------------------------------------
# Analisa sentença e atenções (cacheado por sentença+modelo)
# ------------------------------------------------------------
@st.cache_data
def analyze_attention(sentence, model_name):
    tokenizer, model = load_model_and_tokenizer(model_name)
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    tokenized_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
    offsets = offset_mapping[0].tolist()

    filtered_tokens, filtered_offsets = [], []
    for token, offset in zip(tokenized_text, offsets):
        if token.startswith("##"):
            filtered_tokens[-1] += token.replace("##", "")
        else:
            filtered_tokens.append(token)
            filtered_offsets.append(offset)

    with torch.no_grad():
        outputs = model(**inputs)
        # Move todos os tensores de atenção para CPU imediatamente
        attentions = tuple(a.cpu() for a in outputs.attentions)

    return filtered_tokens, filtered_offsets, attentions

# ------------------------------------------------------------
# Gera mapas de calor de atenção para camadas/cabeças selecionadas
# Renderiza camada a camada para mostrar progresso incremental
# ------------------------------------------------------------
def plot_attn(tokens, attns, selected_layers, selected_heads):
    layer_indices = [l - 1 for l in selected_layers]
    head_indices  = [h - 1 for h in selected_heads]

    n_tokens   = len(tokens)
    tokens_arr = np.array(tokens)
    cmap       = plt.get_cmap("YlGnBu")
    n_heads    = len(head_indices)
    cols       = min(3, n_heads)
    rows       = math.ceil(n_heads / cols)
    records    = []

    progress = st.progress(0, text="Renderizando heatmaps...")

    for step, li in enumerate(layer_indices):
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 5, rows * 5),
            constrained_layout=True,
        )
        axes = np.array(axes).flatten() if n_heads > 1 else [axes]

        for plot_idx, hi in enumerate(head_indices):
            ax   = axes[plot_idx]
            attn = attns[li][0, hi].numpy()[:n_tokens, :n_tokens]

            # imshow renderiza a matriz inteira de uma vez (muito mais rápido que sns.heatmap)
            ax.imshow(attn, cmap=cmap, aspect="auto", vmin=0, vmax=1)
            ax.set_title(f"Layer {li + 1} | Head {hi + 1}", fontsize=8)
            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels(tokens, rotation=45, fontsize=6, ha="right")
            ax.set_yticks(range(n_tokens))
            ax.set_yticklabels(tokens, fontsize=6)

            # Coleta dados vetorizada
            ri, ci = np.meshgrid(range(n_tokens), range(n_tokens), indexing="ij")
            records.append(np.column_stack([
                tokens_arr[ri.ravel()],
                np.full(n_tokens * n_tokens, li + 1),
                np.full(n_tokens * n_tokens, hi + 1),
                tokens_arr[ci.ravel()],
                attn.ravel().round(4),
            ]))

        for idx in range(n_heads, len(axes)):
            fig.delaxes(axes[idx])

        st.pyplot(fig)
        plt.close(fig)
        progress.progress((step + 1) / len(layer_indices),
                          text=f"Camada {li + 1} de {len(layer_indices)} renderizada")

    progress.empty()

    df = pd.DataFrame(
        np.vstack(records),
        columns=["Token", "Layer", "Head", "Attended Token", "Attention Value"],
    )
    df["Layer"] = df["Layer"].astype(int)
    df["Head"] = df["Head"].astype(int)
    df["Attention Value"] = df["Attention Value"].astype(float)
    return df

# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------
st.title("Análise de Atenção — Seleção de Sentença, Padrão e Tokens")

uploaded_file = st.file_uploader(
    "Carregue um arquivo CSV com colunas 'sentence', 'rule', 'token_origem', 'token_destino', 'tokens_to_check':",
    type=["csv"],
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = [
        "sentence",
        "rule",
        "token_origem",
        "token_destino",
        "tokens_to_check",
    ]
    if all(col in df.columns for col in required_cols):
        st.sidebar.header("Configurações de Modelo")

        model_options = {
            "BERT Base Uncased": "google-bert/bert-base-uncased",
            "BERTimbau Base Portuguese Cased": "neuralmind/bert-base-portuguese-cased",
            "mBERT Base Multilingual Uncased": "google-bert/bert-base-multilingual-uncased",
            "mBERT Base Multilingual Cased": "bert-base-multilingual-cased",
        }

        selected_model = st.sidebar.selectbox(
            "Escolha o modelo:", list(model_options.keys())
        )
        model_name = model_options[selected_model]
        tokenizer, model = load_model_and_tokenizer(model_name)

        num_layers = model.config.num_hidden_layers
        num_heads  = model.config.num_attention_heads

        st.sidebar.header("Filtro de Camadas e Cabeças")
        selected_layers = st.sidebar.multiselect(
            "Camadas:", list(range(1, num_layers + 1)), default=list(range(1, num_layers + 1))
        )
        selected_heads = st.sidebar.multiselect(
            "Cabeças:", list(range(1, num_heads + 1)), default=list(range(1, num_heads + 1))
        )

        sentence_options = df["sentence"].unique().tolist()
        selected_sentence = st.selectbox("Escolha a sentença:", sentence_options)

        filtered_df = df[df["sentence"] == selected_sentence]
        pattern_options = filtered_df["rule"].unique().tolist()
        selected_pattern = st.selectbox("Escolha o padrão associado:", pattern_options)

        tokens_df = filtered_df[
            filtered_df["rule"] == selected_pattern
        ][["token_origem", "token_destino"]]

        st.subheader("Sentença e Padrão Selecionados")
        st.write(f"**Sentença:** {selected_sentence}")
        st.write(f"**Padrão:** {selected_pattern}")

        st.subheader("Tokens Governante–Dependente para a Seleção")
        st.dataframe(tokens_df.reset_index(drop=True))

        if st.button("Analisar Atenção da Sentença"):
            if not selected_layers or not selected_heads:
                st.warning("Selecione ao menos uma camada e uma cabeça.")
            else:
                with st.spinner("Calculando atenções..."):
                    tokens, offsets, attentions = analyze_attention(selected_sentence, model_name)

                st.subheader("Mapas de Calor de Atenção")
                with st.spinner("Renderizando heatmaps..."):
                    attention_df = plot_attn(tokens, attentions, selected_layers, selected_heads)

                st.subheader("Tabela Completa de Valores de Atenção")
                st.dataframe(attention_df)
    else:
        st.error(
            "O CSV deve conter as colunas 'sentence', 'rule', 'token_origem', 'token_destino' e 'tokens_to_check'."
        )
else:
    st.warning("Carregue um arquivo CSV para começar.")
