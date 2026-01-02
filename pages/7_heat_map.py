import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(
    page_title="Heatmap de Atenção",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título da página
st.title("Heatmap de Média de Atenção por Camada-Cabeça")

# Carregar o arquivo CSV
uploaded_file = st.file_uploader("Carregue o arquivo CSV para gerar o Heatmap:", type="csv")

if uploaded_file:
    try:
        # Ler o arquivo CSV
        df = pd.read_csv(uploaded_file)
        st.write("📂 **Arquivo carregado com sucesso!**")
        st.dataframe(df.head())

        # Verificar se as colunas obrigatórias existem
        required_columns = {"Layer_Head", "Attention Value", "Layer", "Head", "sentence", "rule"}
        if not required_columns.issubset(df.columns):
            st.error(f"❌ O arquivo não contém todas as colunas necessárias: {required_columns}")
            st.stop()

        # Sidebar: Configuração do Heatmap
        st.sidebar.header("Configurações do Heatmap")

        # Filtro por Regra ou Sentença
        filter_option = st.sidebar.radio("Filtrar por:", ["Regra", "Sentença"])

        if filter_option == "Regra":
            selected_rule = st.sidebar.selectbox("Escolha uma regra:", df["rule"].dropna().unique())
            filtered_df = df[df["rule"] == selected_rule]
        else:
            selected_sentence = st.sidebar.selectbox("Escolha uma sentença:", df["sentence"].dropna().unique())
            filtered_df = df[df["sentence"] == selected_sentence]

        # Remover valores Attention Value = 0 para evitar distorções
        filtered_df = filtered_df[filtered_df["Attention Value"] > 0]

        # Criar colunas separadas para Layer e Head
        filtered_df["Layer"] = filtered_df["Layer_Head"].apply(lambda x: int(x.split("_")[0]))
        filtered_df["Head"] = filtered_df["Layer_Head"].apply(lambda x: int(x.split("_")[1]))

        # Calcular média da atenção por camada e cabeça
        heatmap_data = (
            filtered_df.groupby(["Layer", "Head"])["Attention Value"]
            .mean()
            .reset_index()
            .pivot(index="Layer", columns="Head", values="Attention Value")
        )

        # Sidebar: Escolher intervalo de camadas para exibir
        min_layer = int(heatmap_data.index.min())
        max_layer = int(heatmap_data.index.max())

        selected_layers = st.sidebar.slider(
            "Selecionar intervalo de camadas:",
            min_value=min_layer,
            max_value=max_layer,
            value=(min_layer, max_layer)
        )

        heatmap_data = heatmap_data.loc[selected_layers[0]:selected_layers[1]]

        # **Plotando Heatmap usando Plotly**
        st.subheader("Mapa de Calor das Médias de Atenção por Camada-Cabeça")

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="Blues",
            hoverongaps=False
        ))

        fig.update_layout(
            title="Heatmap de Média de Atenção por Camada-Cabeça",
            xaxis_title="Cabeça de Atenção",
            yaxis_title="Camada",
            width=800,
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {e}")

else:
    st.info("📥 **Por favor, carregue um arquivo CSV para começar.**")
