import streamlit as st

# ------------------------------------------------------------
# Configuração geral
# ------------------------------------------------------------
st.set_page_config(
    page_title="Attention Analysis Hub",
    layout="wide"
)

# ------------------------------------------------------------
# Página inicial (Home)
# ------------------------------------------------------------
st.title("Attention Analysis Hub")

st.markdown(
    """
Este aplicativo reúne diferentes módulos de análise de atenção em modelos
Transformer aplicados ao português brasileiro.

A navegação segue o padrão **multipage nativo do Streamlit**, permitindo acesso
direto a cada experimento por meio do menu lateral.
"""
)

# ------------------------------------------------------------
# Descrição estruturada dos módulos
# ------------------------------------------------------------
st.subheader("Módulos disponíveis")

modules = {
    "Classificar Sentenças": "1_classificar_sentencas.py",
    "Análise de Atenção BERT/RoBERTa (Plot 1)": "2_streamlit_plot_bosque_fast_tokenizer_1.py",
    "Análise Focada com CSV (Plot 2)": "3_streamlit_plot_bosque_fast_tokenizer_2.py",
    "Mapas de Calor com Tokens Especiais (Plot 3)": "4_streamlit_plot_bosque_fast_tokenizer_3.py",
    "Análise de Regras BERT (Regras)": "5_streamlit_analisar_regras_fast_tokenizer_1.py",
    "Treemap Interativo": "6_tree_map.py",
}

for title, filename in modules.items():
    st.markdown(f"- **{title}** (`pages/{filename}`)")

# ------------------------------------------------------------
# Nota técnica
# ------------------------------------------------------------
st.info(
    """
Use o menu lateral à esquerda para acessar cada módulo.

Cada item corresponde a um script independente localizado na pasta `pages/`,
executado no mesmo processo Streamlit, conforme o padrão oficial da plataforma.
"""
)
