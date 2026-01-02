import streamlit as st
import subprocess

# Configuração da página do hub
st.set_page_config(
    page_title="Analisador Transformer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Título do hub
st.title("Explorador de Cabeças de Atenção em Transformers")
st.write("Bem-vindo ao Explorador de Cabeças de Atenção: Descubra como os Transformers enxergam o mundo através de suas atenções!")

# Mapeamento dos arquivos carregados para nomes de aplicações
apps = {
    "Classificar Sentenças": "1_classificar_sentencas.py",
    "Análise de Atenção BERT/RoBERTa (Plot 1)": "2_streamlit_plot_bosque_fast_tokenizer_1.py",
    "Análise Focada com CSV (Plot 2)": "3_streamlit_plot_bosque_fast_tokenizer_2.py",
    "Mapas de Calor com Tokens Especiais (Plot 3)": "4_streamlit_plot_bosque_fast_tokenizer_3.py",
    "Análise de Regras BERT (Regras)": "5_streamlit_analisar_regras_fast_tokenizer_1.py",
    "Treemap Interativo": "6_tree_map.py",
    
}

# Menu na barra lateral à direita
st.sidebar.title("Menu de Aplicações")
st.sidebar.write("Selecione uma aplicação abaixo para executá-la.")

# Seleção da aplicação na barra lateral
app_name = st.sidebar.radio("Escolha uma aplicação:", list(apps.keys()))

# Botão para iniciar a aplicação selecionada
if st.sidebar.button("Executar Aplicação"):
    app_file = apps[app_name]
    try:
        st.sidebar.success(f"Iniciando a aplicação: {app_name}")
        # Comando para iniciar o Streamlit em uma nova aba do navegador
        subprocess.Popen(["streamlit", "run", app_file])
    except Exception as e:
        st.sidebar.error(f"Erro ao iniciar a aplicação: {e}")

st.sidebar.info("A aplicação será carregada em uma nova aba do navegador.")

