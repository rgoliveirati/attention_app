import streamlit as st
import pandas as pd
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="Treemap Interativo",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título da página
st.title("Treemap Interativo")

# Carregar o arquivo CSV
uploaded_file = st.file_uploader("Carregue o arquivo CSV para gerar o Treemap:", type="csv")

if uploaded_file:
    # Ler o arquivo CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Dados carregados com sucesso!")
        st.dataframe(df)

        st.sidebar.header("Configurações do Treemap")
        
        # Opção para incluir ou excluir CLS e SEP
        include_cls_sep = st.sidebar.radio(
            "Incluir CLS e SEP no DataFrame?",
            options=["Sim", "Não"],
            index=0
        )

        if include_cls_sep == "Não":
            # Filtrar valores CLS e SEP das colunas 'Token' e 'Attended Token'
            df = df[~df["Token"].isin(["[CLS]", "[SEP]"]) & ~df["Attended Token"].isin(["[CLS]", "[SEP]"])]
            st.write("Valores CLS e SEP foram removidos do DataFrame:")
            st.dataframe(df)

        # Selecionar colunas para o Treemap
        available_columns = df.columns.tolist()
        
        label_column = st.sidebar.selectbox("Coluna de Labels:", available_columns, index=0)
        parent_column = st.sidebar.selectbox("Coluna de Pais:", available_columns, index=1)
        value_column = st.sidebar.selectbox("Coluna de Valores:", available_columns, index=2)
        
        # Verificar se as colunas selecionadas são válidas
        if st.sidebar.button("Gerar Treemap"):
            st.write("Gerando o Treemap com as configurações selecionadas...")
            
            # Gerar o Treemap
            fig = px.treemap(
                df,
                path=[parent_column, label_column],
                values=value_column,
                title="Treemap Interativo",
                color=value_column,
                color_continuous_scale="Viridis",
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("Por favor, carregue um arquivo CSV para começar.")
