# -*- coding: utf-8 -*-
# ============================================================
# Análise de Atenção — Seleção de Sentença, Padrão e Tokens
# ============================================================

import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizerFast, BertModel
import math

# ------------------------------------------------------------
# Configuração da página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Análise por Sentença, Padrão e Tokens",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Detecta dispositivo: GPU se disponível, caso contrário CPU
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.text(f"Dispositivo: {device}")

# ------------------------------------------------------------
# Carrega modelo e tokenizador
# ------------------------------------------------------------
def load_model_and_tokenizer(model_name):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model = model.to(device)
    return tokenizer, model

# ------------------------------------------------------------
# Analisa sentença e atenções
# ------------------------------------------------------------
def analyze_attention(sentence, tokenizer, model):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    # Envia tensores relevantes ao dispositivo
    inputs = {k: v.to(device) if k != "offset_mapping" else v for k, v in inputs.items()}

    tokenized_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
    offsets = inputs["offset_mapping"][0].tolist()

    filtered_tokens, filtered_offsets = [], []
    for token, offset in zip(tokenized_text, offsets):
        if token.startswith("##"):
            filtered_tokens[-1] += token.replace("##", "")
        else:
            filtered_tokens.append(token)
            filtered_offsets.append(offset)

    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})
        attentions = outputs.attentions  # tensores no dispositivo

    return filtered_tokens, filtered_offsets, attentions

# ------------------------------------------------------------
# Gera mapas de calor de atenção
# ------------------------------------------------------------
def plot_attn(tokens, attns):
    num_layers = len(attns)
    num_heads = attns[0].size(1)
    total_plots = num_layers * num_heads
    cols = 3
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    data = []
    for layer in range(num_layers):
        for head in range(num_heads):
            idx = layer * num_heads + head
            ax = axes[idx]

            attn = attns[layer][0, head].detach().cpu().numpy()  # move para CPU antes de numpy
            n_tokens = len(tokens)
            filtered_attn = attn[:n_tokens, :n_tokens]

            sns.heatmap(
                filtered_attn,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="YlGnBu",
                cbar=False,
                annot=False,
                ax=ax,
            )
            ax.set_title(f"Layer {layer + 1} | Head {head + 1}", fontsize=8)
            ax.tick_params(axis="x", rotation=45, labelsize=6)
            ax.tick_params(axis="y", rotation=0, labelsize=6)

            for i, token in enumerate(tokens):
                for j, attended_token in enumerate(tokens):
                    data.append(
                        [
                            token,
                            layer + 1,
                            head + 1,
                            attended_token,
                            round(filtered_attn[i][j], 4),
                        ]
                    )

    for idx in range(total_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    st.pyplot(fig)

    df = pd.DataFrame(
        data,
        columns=[
            "Token",
            "Layer",
            "Head",
            "Attended Token",
            "Attention Value",
        ],
    )
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
        tokenizer, model = load_model_and_tokenizer(model_options[selected_model])

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
            tokens, offsets, attentions = analyze_attention(
                selected_sentence, tokenizer, model
            )

            st.subheader("Mapas de Calor de Atenção")
            attention_df = plot_attn(tokens, attentions)

            st.subheader("Tabela Completa de Valores de Atenção")
            st.dataframe(attention_df)
    else:
        st.error(
            "O CSV deve conter as colunas 'sentence', 'rule', 'token_origem', 'token_destino' e 'tokens_to_check'."
        )
else:
    st.warning("Carregue um arquivo CSV para começar.")

