import streamlit as st
import pandas as pd
from transformers import BertTokenizerFast, BertModel
import torch

st.set_page_config(
    page_title="Análise de Sentenças e Padrões",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_model_and_tokenizer(model_name):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

def analyze_attention(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True, return_offsets_mapping=True)
    tokenized_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    offsets = inputs['offset_mapping'][0].tolist()

    filtered_tokens = []
    filtered_offsets = []

    for token, offset in zip(tokenized_text, offsets):
        if token.startswith("##"):  
            filtered_tokens[-1] += token.replace("##", "")
        else:
            filtered_tokens.append(token)
            filtered_offsets.append(offset)

    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
        attentions = outputs.attentions

    return filtered_tokens, filtered_offsets, attentions

def create_attention_df(tokens, offsets, attentions):
    data = []
    num_layers = len(attentions)
    num_heads = attentions[0].size(1)

    for layer in range(num_layers):
        for head in range(num_heads):
            attn = attentions[layer][0, head].detach().numpy()
            for i, token in enumerate(tokens):
                for j, attended_token in enumerate(tokens):
                    data.append([token, layer + 1, head + 1, attended_token, round(attn[i][j], 4)])

    df = pd.DataFrame(data, columns=['Token', 'Layer', 'Head', 'Attended Token', 'Attention Value'])
    df["Layer_Head"] = df["Layer"].astype(str) + "_" + df["Head"].astype(str)
    return df

st.title('Análise de Atenção — Padrões das Sentenças')

uploaded_file = st.file_uploader("Carregue um arquivo CSV com 'sentence', 'rule', 'tokens_to_check':", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in ["sentence", "rule", "tokens_to_check"]):
        st.error("O arquivo deve conter as colunas: 'sentence', 'rule' e 'tokens_to_check'.")
    else:
        model_options = {
            "BERT Base Uncased": 'bert-base-uncased',
            "BERT Base Cased": 'bert-base-cased',
            "BERTimbau Base Portuguese Cased": 'neuralmind/bert-base-portuguese-cased',
            "mBERT Base Multilingual Uncased": 'bert-base-multilingual-uncased',
            "mBERT Base Multilingual Cased": 'bert-base-multilingual-cased',
        }

        selected_model = st.selectbox('Escolha o modelo:', list(model_options.keys()))
        model_name = model_options[selected_model]
        tokenizer, model = load_model_and_tokenizer(model_name)

        sentence_options = df["sentence"].unique().tolist()
        selected_sentence = st.selectbox("Escolha a sentença:", sentence_options)

        filtered_df = df[df["sentence"] == selected_sentence]
        pattern_options = filtered_df["rule"].unique().tolist()
        selected_pattern = st.selectbox("Escolha o padrão associado:", pattern_options)

        tokens_to_check = filtered_df[filtered_df["rule"] == selected_pattern]["tokens_to_check"].tolist()
        st.subheader("Sentença e Padrão Selecionados")
        st.write(f"**Sentença:** {selected_sentence}")
        st.write(f"**Padrão:** {selected_pattern}")
        st.write("**Tokens a Checar:**")
        st.write(tokens_to_check)

        if st.button("Analisar Sentença Selecionada"):
            tokens, offsets, attentions = analyze_attention(selected_sentence, tokenizer, model)
            attention_df = create_attention_df(tokens, offsets, attentions)
            attention_df["sentence"] = selected_sentence
            attention_df["rule"] = selected_pattern
            st.subheader("Resultados da Análise de Atenção")
            st.dataframe(attention_df)

        num_sentences = st.number_input(
            "Quantas sentenças deseja processar (todos os seus padrões)?", 
            min_value=1, 
            max_value=len(df["sentence"].unique()), 
            value=3, 
            step=1
        )

        if st.button("Analisar Todas as Sentenças Selecionadas"):
            results = []

            selected_sentences = df["sentence"].drop_duplicates().head(num_sentences).tolist()
            subset_df = df[df["sentence"].isin(selected_sentences)]

            for _, row in subset_df.iterrows():
                sentence = row["sentence"]
                rule = row["rule"]
                tokens, offsets, attentions = analyze_attention(sentence, tokenizer, model)
                attention_df = create_attention_df(tokens, offsets, attentions)
                attention_df["sentence"] = sentence
                attention_df["rule"] = rule
                results.append(attention_df)

            results_df = pd.concat(results, ignore_index=True)
            st.success(f"Análise concluída! Foram processados {len(selected_sentences)} sentenças e {len(results_df['sentence'].unique())} sentenças distintas no total.")
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Baixar Resultados da Análise",
                data=csv,
                file_name=f"analise_sentencas_todos_padroes.csv",
                mime="text/csv",
            )
else:
    st.info("Por favor, carregue um arquivo CSV para começar.")
