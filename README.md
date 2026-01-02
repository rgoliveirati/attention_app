# Attention Analysis Hub

Este repositório disponibiliza uma aplicação **Streamlit multipágina** para análise e visualização de **mecanismos de atenção em modelos Transformer** aplicados ao **português brasileiro**, com foco em inspeção atencional, padrões sintáticos e regras linguísticas.

O projeto consolida diferentes experimentos em um único ambiente interativo, seguindo o **padrão oficial de multipage do Streamlit**, adequado para execução local e deploy em ambiente gerenciado (Streamlit Community Cloud).

---

## Estrutura do Repositório

```
attention/
├── hub.py
├── pages/
│   ├── 1_classificar_sentencas.py
│   ├── 2_streamlit_plot_bosque_fast_tokenizer_1.py
│   ├── 3_streamlit_plot_bosque_fast_tokenizer_2.py
│   ├── 4_streamlit_plot_bosque_fast_tokenizer_3.py
│   ├── 5_streamlit_analisar_regras_fast_tokenizer_1.py
│   ├── 6_tree_map.py
│
├── data/
│   └── pt_bosque-ud-train.conllu
│
├── requirements.txt
└── README.md
```

* `hub.py`
  Página inicial (Home) da aplicação.
* `pages/`
  Conjunto de aplicações Streamlit independentes.
* `data/`
  Arquivos de dados linguísticos necessários para alguns módulos.
* `requirements.txt`
  Dependências Python.

---

## ⚠️ Dependência de Dados Linguísticos (Obrigatória)

### Corpus UD Portuguese-Bosque

Para que o módulo **Classificar Sentenças** funcione corretamente, é **obrigatória** a presença do arquivo:

```
pt_bosque-ud-train.conllu
```

Este arquivo pertence ao corpus **UD Portuguese-Bosque** (Universal Dependencies) e é utilizado como base para:

* leitura de sentenças anotadas,
* acesso a dependências sintáticas,
* inicialização e classificação estrutural das sentenças.

### Onde colocar o arquivo

O arquivo deve estar disponível no seguinte caminho relativo ao projeto:

```
data/pt_bosque-ud-train.conllu
```

Caso o arquivo não esteja presente, o módulo de classificação **não será inicializado**.

### Obtenção do corpus

O corpus pode ser obtido a partir do repositório oficial do Universal Dependencies:

```
https://github.com/UniversalDependencies/UD_Portuguese-Bosque
```

Após o download, copie o arquivo `pt_bosque-ud-train.conllu` para a pasta `data/`.

---

## Módulos Disponíveis

* **Classificar Sentenças**
  Classificação de sentenças com base em estruturas sintáticas do corpus UD Portuguese-Bosque.
* **Análise de Atenção BERT/RoBERTa (Plot 1)**
  Visualização de pesos de atenção com tokenização rápida.
* **Análise Focada com CSV (Plot 2)**
  Inspeção direcionada de padrões de atenção a partir de dados externos.
* **Mapas de Calor com Tokens Especiais (Plot 3)**
  Heatmaps de atenção incluindo tokens especiais.
* **Análise de Regras BERT (Regras)**
  Avaliação de regras linguísticas e padrões estruturais.
* **Treemap Interativo**
  Visualização hierárquica de padrões e distribuições de atenção.

---

## Requisitos

* Python 3.9 ou superior
* Dependências listadas em `requirements.txt`, incluindo:

  * `streamlit`
  * `transformers`
  * `torch`
  * `pandas`
  * `numpy`
  * `plotly`
  * `seaborn`
  * `conllu`

---

## Execução Local

```bash
git clone https://github.com/<usuario>/attention.git
cd attention
pip install -r requirements.txt
streamlit run hub.py
```

---

## Deploy no Streamlit Community Cloud

* Repository: `attention`
* Branch: `main`
* Main file path: `hub.py`

**Certifique-se de que o arquivo `data/pt_bosque-ud-train.conllu` esteja versionado ou acessível no ambiente de deploy.**

---

## Considerações Técnicas

* Cada arquivo em `pages/` é executado no mesmo processo Streamlit.
* O projeto segue o modelo **um app / um processo**, sem uso de `subprocess`.
* Recomenda-se o uso de `@st.cache_resource` para carregamento de modelos Transformer.
* O corpus UD é tratado como **dependência de dados**, não como dependência de código.

---

## Contexto Científico

Este repositório integra experimentos voltados à **interpretabilidade de mecanismos de atenção em Transformers**, com ênfase em:

* alinhamento entre atenção e dependências sintáticas,
* análise de padrões gramaticais segundo Universal Dependencies,
* visualização estruturada de pesos atencionais,
* aplicação ao português brasileiro.
