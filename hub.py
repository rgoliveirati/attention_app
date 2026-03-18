import streamlit as st

# ============================================================
# Configuração geral
# ============================================================
st.set_page_config(
    page_title="Attention Analysis Hub",
    layout="wide"
)

# ============================================================
# Estilo
# ============================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #666;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 1.2rem;
            margin-bottom: 0.8rem;
        }

        .hub-card {
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 16px;
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 1rem;
            background: rgba(250,250,250,0.55);
        }

        .hub-card-title {
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .hub-card-text {
            color: #555;
            font-size: 0.95rem;
            min-height: 60px;
        }

        .hub-badge {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: #eef3ff;
            color: #1f4fbf;
            margin-bottom: 0.6rem;
        }

        .hub-note {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            background: rgba(240,245,255,0.75);
            border: 1px solid rgba(100,130,220,0.18);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .hub-step {
            border-left: 4px solid #4f46e5;
            padding: 0.55rem 0.9rem;
            margin-bottom: 0.8rem;
            background: rgba(79,70,229,0.05);
            border-radius: 0 10px 10px 0;
        }

        .small-muted {
            color: #777;
            font-size: 0.85rem;
        }

        .open-link a {
            text-decoration: none;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Metadados dos módulos
# ============================================================
MODULES = [
    {
        "file": "pages/0_informacoes.py",
        "title": "Informações",
        "pathname": "informacoes",
        "category": "Guia",
        "description": "Página introdutória com explicações sobre o fluxo de uso, entradas esperadas e finalidade de cada módulo."
    },
    {
        "file": "pages/1_classificar_sentencas.py",
        "title": "Classificar Sentenças",
        "pathname": "classificar_sentencas",
        "category": "Preparação",
        "description": "Lê arquivos .conllu, identifica padrões gramaticais e gera dados estruturados para análises posteriores."
    },
    {
        "file": "pages/2_streamlit_plot_bosque_fast_tokenizer_1.py",
        "title": "Análise de Atenção BERT/RoBERTa (Plot 1)",
        "pathname": "streamlit_plot_bosque_fast_tokenizer_1",
        "category": "Análise",
        "description": "Mostra uma visão geral da atenção entre tokens em modelos Transformer a partir de sentenças e regras."
    },
    {
        "file": "pages/3_streamlit_plot_bosque_fast_tokenizer_2.py",
        "title": "Análise Focada com CSV (Plot 2)",
        "pathname": "streamlit_plot_bosque_fast_tokenizer_2",
        "category": "Análise",
        "description": "Permite destacar palavras de interesse e inspecionar com mais foco o comportamento da atenção."
    },
    {
        "file": "pages/4_streamlit_plot_bosque_fast_tokenizer_3.py",
        "title": "Mapas de Calor com Tokens Especiais (Plot 3)",
        "pathname": "streamlit_plot_bosque_fast_tokenizer_3",
        "category": "Análise Avançada",
        "description": "Exibe mapas de calor detalhados e relações específicas entre pares de tokens relevantes."
    },
    {
        "file": "pages/5_streamlit_analisar_regras_fast_tokenizer_1.py",
        "title": "Análise de Regras BERT (Regras)",
        "pathname": "streamlit_analisar_regras_fast_tokenizer_1",
        "category": "Processamento",
        "description": "Executa análises orientadas por regras e facilita geração de resultados tabulares para exportação."
    },
    {
        "file": "pages/6_tree_map.py",
        "title": "Treemap Interativo",
        "pathname": "tree_map",
        "category": "Visualização",
        "description": "Cria visualização hierárquica em treemap para exploração resumida de distribuições e agrupamentos."
    },
    {
        "file": "pages/7_heat_map.py",
        "title": "Heat Map",
        "pathname": "heat_map",
        "category": "Visualização",
        "description": "Gera mapa de calor por camada e cabeça para identificar padrões globais de atenção."
    },
]

# ============================================================
# Helper para abrir em nova aba
# ============================================================
def page_url(pathname: str) -> str:
    return f"/{pathname}"

# ============================================================
# Home
# ============================================================
def home():
    st.markdown('<div class="main-title">Attention Analysis Hub</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="subtitle">
            Ambiente unificado para análise de atenção em modelos Transformer
            aplicados ao português brasileiro.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown(
            """
            Este aplicativo reúne módulos para preparação de dados, análise de atenção,
            inspeção orientada por regras e visualizações agregadas.

            A navegação principal está no menu lateral.  
            Na área abaixo, cada módulo também pode ser aberto em uma nova aba.
            """
        )

    with col_b:
        st.markdown(
            """
            <div class="hub-note">
                <strong>Primeiro acesso?</strong><br>
                Comece pela página <strong>Informações</strong> para entender
                o fluxo ideal de uso e o formato esperado dos arquivos.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Fluxo recomendado</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hub-step"><strong>1. Informações</strong><br>Entenda o papel de cada página e o tipo de arquivo necessário.</div>
        <div class="hub-step"><strong>2. Classificar Sentenças</strong><br>Comece aqui se você possui arquivo <code>.conllu</code>.</div>
        <div class="hub-step"><strong>3. Análises de Atenção</strong><br>Use Plot 1, Plot 2, Plot 3 e Regras para inspecionar os resultados.</div>
        <div class="hub-step"><strong>4. Visualizações Finais</strong><br>Use Treemap e Heat Map para síntese visual dos dados.</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Módulos disponíveis</div>', unsafe_allow_html=True)

    for module in MODULES:
        with st.container():
            col1, col2 = st.columns([5.3, 1.2])

            with col1:
                st.markdown(
                    f"""
                    <div class="hub-card">
                        <div class="hub-badge">{module["category"]}</div>
                        <div class="hub-card-title">{module["title"]}</div>
                        <div class="hub-card-text">{module["description"]}</div>
                        <div class="small-muted"><code>{module["file"]}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.page_link(module["file"], label="Abrir aqui", icon="📄")
                st.markdown(
                    f"""
                    <div class="open-link">
                        <a href="{page_url(module["pathname"])}" target="_blank">🔗 Abrir em nova aba</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.info(
        """
        O menu lateral navega na mesma aba.
        O link **Abrir em nova aba** cria outra sessão do Streamlit no navegador.
        """
    )

# ============================================================
# Navegação
# ============================================================
pages = [
    st.Page(home, title="Attention Analysis Hub", default=True)
]

for module in MODULES:
    pages.append(
        st.Page(
            module["file"],
            title=module["title"],
            url_path=module["pathname"],
        )
    )

pg = st.navigation(pages)
pg.run()