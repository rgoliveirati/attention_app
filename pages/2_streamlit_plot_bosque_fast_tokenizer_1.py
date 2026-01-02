import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast, BertModel, RobertaTokenizerFast, RobertaModel
import numpy as np

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Atenção Ampla",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Função para carregar o tokenizer e o modelo
def load_model_and_tokenizer(model_name):
    if "roberta" in model_name.lower():
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name, output_attentions=True)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

# Função para analisar a sentença e atenções
def analyze_attention(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True, return_offsets_mapping=True)
    tokenized_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    offsets = inputs['offset_mapping'][0].tolist()

    # Remover tokens especiais [CLS] e [SEP] e reconstruir palavras
    filtered_tokens = []
    filtered_offsets = []

    for token, offset in zip(tokenized_text, offsets):
    #     if token in ['[CLS]', '[SEP]']:
    #     continue  # Ignorar tokens especiais

        if token.startswith("##"):  
            filtered_tokens[-1] += token.replace("##", "")  # Junta ao token anterior
        else:
            filtered_tokens.append(token)
            filtered_offsets.append(offset)

    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
        attentions = outputs.attentions
    
    return filtered_tokens, filtered_offsets, attentions

# Função para plotar as atenções
def plot_attn(tokens, attns, heads):
    width = 3
    example_sep = 3
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
        attn = attns[layer][0, head].detach().numpy()
        words = tokens  
        n_words = len(words)

        yoffset = 1
        xoffset = ei * width * example_sep

        plt.title(f"Layer {layer + 1}, Head {head + 1}")
        plt.axis("off")

        for position, word in enumerate(words):
            plt.text(xoffset + 0, yoffset - position * word_height, word, ha="right", va="center")
            plt.text(xoffset + width, yoffset - position * word_height, word, ha="left", va="center")

        for i in range(n_words):
            for j in range(n_words):
                plt.plot([xoffset + pad, xoffset + width - pad],
                         [yoffset - word_height * i, yoffset - word_height * j],
                         color="blue", linewidth=1, alpha=attn[i, j])
        with cols[count]:
            st.pyplot(fig)

# Interface do Streamlit
st.title('Análise de Atenção Ampla')

# Escolher o modelo e o tokenizador
model_options = {
    "BERT Base Uncased": 'bert-base-uncased',
    "BERTimbau Base Portuguese Cased": 'neuralmind/bert-base-portuguese-cased',
    "mBERT Base Multilingual Uncased": 'bert-base-multilingual-uncased',
    "mBERT Base Multilingual Cased": 'bert-base-multilingual-cased',
    "RoBERTa Base": 'roberta-base'
}

selected_model = st.selectbox('Escolha o modelo:', list(model_options.keys()))
model_name = model_options[selected_model]
tokenizer, model = load_model_and_tokenizer(model_name)

# Carregar o dataset
uploaded_file = st.file_uploader("Carregue o arquivo CSV com sentenças:", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "sentence" in df.columns and "rule" in df.columns:
        sentence_options = df["sentence"].tolist()
        selected_sentence = st.selectbox('Escolha a frase:', sentence_options)
        rule = df[df["sentence"] == selected_sentence]["rule"].values[0]

        st.subheader("Informações da Sentença Selecionada")
        st.write(f"**Regra Gramatical:** {rule}")

        layers = st.slider('Escolha a camada:', 1, 12, 1)
        heads_per_layer = st.slider('Escolha a cabeça:', 1, 12, 1)

        if st.button('Analisar'):
            tokens, _, attentions = analyze_attention(selected_sentence, tokenizer, model)  # Ignorando offsets
            heads = [(layer, head) for layer in range(layers) for head in range(heads_per_layer)]
            st.divider()
            plot_attn(tokens, attentions, heads)
    else:
        st.error("O arquivo CSV deve conter as colunas 'sentence' e 'rule'.")
else:
    st.warning("Por favor, carregue um arquivo CSV para continuar.")
