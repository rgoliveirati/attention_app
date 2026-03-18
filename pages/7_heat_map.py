import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# # Configuração da página
# st.set_page_config(
#     page_title="Heatmap de Atenção",
#     page_icon="🔥",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

st.title("Heatmap de Média de Atenção por Camada-Cabeça")


# =========================
# Funções auxiliares
# =========================
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    """
    Lê o CSV uma única vez por arquivo carregado.
    """
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def preprocess_df(df):
    """
    Faz o pré-processamento uma única vez:
    - valida colunas
    - cria Layer e Head a partir de Layer_Head se necessário
    - remove Attention Value <= 0
    """
    df = df.copy()

    required_base = {"Attention Value", "sentence", "rule"}
    if not required_base.issubset(df.columns):
        missing = required_base - set(df.columns)
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    has_layer_head = "Layer_Head" in df.columns
    has_layer_and_head = {"Layer", "Head"}.issubset(df.columns)

    if not has_layer_head and not has_layer_and_head:
        raise ValueError(
            "O arquivo precisa ter 'Layer_Head' ou então as colunas 'Layer' e 'Head'."
        )

    # Se não houver Layer e Head, cria a partir de Layer_Head
    if not has_layer_and_head:
        try:
            split_vals = df["Layer_Head"].astype(str).str.split("_", expand=True)
            df["Layer"] = split_vals[0].astype(int)
            df["Head"] = split_vals[1].astype(int)
        except Exception as e:
            raise ValueError(f"Não foi possível extrair Layer e Head de 'Layer_Head': {e}")

    else:
        df["Layer"] = pd.to_numeric(df["Layer"], errors="coerce")
        df["Head"] = pd.to_numeric(df["Head"], errors="coerce")

        # Se existir Layer_Head inconsistente, ignoramos e usamos Layer/Head explícitos
        df = df.dropna(subset=["Layer", "Head"])
        df["Layer"] = df["Layer"].astype(int)
        df["Head"] = df["Head"].astype(int)

    df["Attention Value"] = pd.to_numeric(df["Attention Value"], errors="coerce")
    df = df.dropna(subset=["Attention Value", "Layer", "Head", "sentence", "rule"])

    # Remove zeros e negativos para evitar distorções
    df = df[df["Attention Value"] > 0].copy()

    return df


def build_heatmap_matrix(df_filtered):
    """
    Agrega por Layer x Head e devolve a matriz do heatmap.
    """
    heatmap_data = (
        df_filtered.groupby(["Layer", "Head"], as_index=False)["Attention Value"]
        .mean()
        .pivot(index="Layer", columns="Head", values="Attention Value")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return heatmap_data


def plot_heatmap(heatmap_data, title):
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="Blues",
            hoverongaps=False,
            hovertemplate=(
                "Camada: %{y}<br>"
                "Cabeça: %{x}<br>"
                "Atenção média: %{z:.6f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Cabeça de Atenção",
        yaxis_title="Camada",
        height=700,
    )
    return fig


# =========================
# Upload
# =========================
uploaded_file = st.file_uploader(
    "Carregue o arquivo CSV para gerar o Heatmap:",
    type="csv"
)

if uploaded_file:
    try:
        df_raw = load_csv(uploaded_file)
        df = preprocess_df(df_raw)

        st.write("📂 **Arquivo carregado com sucesso!**")
        with st.expander("Visualizar amostra dos dados processados"):
            st.dataframe(df.head())

        if df.empty:
            st.warning("Após o pré-processamento, não restaram dados válidos para plotagem.")
            st.stop()

        # =========================
        # Sidebar
        # =========================
        st.sidebar.header("Configurações do Heatmap")

        modo = st.sidebar.radio(
            "Modo de análise:",
            [
                "Heatmap por sentença",
                "Heatmap médio por regra",
            ]
        )

        filtered_df = df.copy()
        subtitle = ""

        if modo == "Heatmap por sentença":
            selected_sentence = st.sidebar.selectbox(
                "Escolha uma sentença:",
                sorted(df["sentence"].dropna().unique())
            )
            filtered_df = df[df["sentence"] == selected_sentence].copy()
            subtitle = f"Sentença: {selected_sentence}"

            # filtro opcional adicional por regra dentro da sentença
            regras_disponiveis = sorted(filtered_df["rule"].dropna().unique())
            if len(regras_disponiveis) > 1:
                selected_rule_inside = st.sidebar.selectbox(
                    "Filtrar também por regra dentro da sentença:",
                    ["Todas"] + regras_disponiveis
                )
                if selected_rule_inside != "Todas":
                    filtered_df = filtered_df[filtered_df["rule"] == selected_rule_inside].copy()
                    subtitle += f" | Regra: {selected_rule_inside}"

        else:
            selected_rule = st.sidebar.selectbox(
                "Escolha uma regra:",
                sorted(df["rule"].dropna().unique())
            )
            filtered_df = df[df["rule"] == selected_rule].copy()
            subtitle = f"Regra: {selected_rule}"

            qtd_sentencas = filtered_df["sentence"].nunique()
            st.sidebar.caption(f"Sentenças usadas na média: {qtd_sentencas}")

        if filtered_df.empty:
            st.warning("Nenhum dado encontrado para o filtro selecionado.")
            st.stop()

        heatmap_data = build_heatmap_matrix(filtered_df)

        if heatmap_data.empty:
            st.warning("Não foi possível montar a matriz do heatmap para o filtro selecionado.")
            st.stop()

        min_layer = int(heatmap_data.index.min())
        max_layer = int(heatmap_data.index.max())

        selected_layers = st.sidebar.slider(
            "Selecionar intervalo de camadas:",
            min_value=min_layer,
            max_value=max_layer,
            value=(min_layer, max_layer)
        )

        heatmap_data = heatmap_data.loc[selected_layers[0]:selected_layers[1]]

        st.subheader("Mapa de Calor das Médias de Atenção por Camada-Cabeça")
        st.caption(subtitle)

        fig = plot_heatmap(
            heatmap_data,
            title="Heatmap de Média de Atenção por Camada-Cabeça"
        )
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # Resumo numérico
        # =========================
        st.markdown("### Resumo")
        c1, c2, c3 = st.columns(3)
        c1.metric("Linhas filtradas", len(filtered_df))
        c2.metric("Sentenças únicas", filtered_df["sentence"].nunique())
        c3.metric("Regras únicas", filtered_df["rule"].nunique())

        with st.expander("Ver matriz Layer × Head"):
            st.dataframe(heatmap_data)

    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {e}")

else:
    st.info("📥 **Por favor, carregue um arquivo CSV para começar.**")