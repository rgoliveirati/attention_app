import streamlit as st

st.title("Informações e Guia de Uso")

st.markdown(
    """
Esta página explica, de forma simples, para que serve cada módulo do sistema,
qual arquivo deve ser usado em cada um e em que ordem o uso costuma fazer mais sentido.
"""
)

st.subheader("Fluxo recomendado")

st.markdown(
    """
Para a maioria dos casos, o uso mais natural do sistema é este:

1. **Classificar Sentenças**  
   Quando você tem um arquivo `.conllu` e quer identificar padrões gramaticais.

2. **Análises de Atenção (Plots e Regras)**  
   Quando você já tem sentenças organizadas em CSV e deseja investigar a atenção do modelo.

3. **Treemap e Heat Map**  
   Quando você quer visualizar resultados agregados de análises já realizadas.
"""
)

st.divider()

with st.expander("1. Classificar Sentenças", expanded=True):
    st.markdown(
        """
**Para que serve**  
Lê um arquivo `.conllu` anotado em Universal Dependencies e identifica padrões gramaticais
nas sentenças.

**Entrada esperada**  
- Arquivo `.conllu`

**O que esta página faz**  
- carrega sentenças do corpus anotado;
- identifica padrões gramaticais;
- extrai relações relevantes entre governante e dependente;
- gera uma tabela estruturada;
- permite baixar um CSV para uso posterior.

**Quando usar**  
Use esta página no início do processo, quando você ainda está partindo do corpus linguístico anotado.
"""
    )

with st.expander("2. Análise de Atenção BERT/RoBERTa (Plot 1)"):
    st.markdown(
        """
**Para que serve**  
Exibe a atenção de sentenças em modelos Transformer, com foco em uma visualização mais ampla
das conexões entre tokens.

**Entrada esperada**  
CSV com pelo menos estas colunas:
- `sentence`
- `rule`

**O que esta página faz**  
- permite escolher o modelo;
- permite selecionar uma sentença;
- mostra o padrão gramatical associado;
- exibe gráficos e análises de atenção entre tokens.

**Quando usar**  
Use quando quiser uma visão geral da atenção da sentença no modelo.
"""
    )

with st.expander("3. Análise Focada com CSV (Plot 2)"):
    st.markdown(
        """
**Para que serve**  
Permite analisar a atenção com foco mais dirigido, destacando palavras ou trechos de interesse.

**Entrada esperada**  
CSV com pelo menos estas colunas:
- `sentence`
- `rule`

**O que esta página faz**  
- permite selecionar sentença e padrão;
- permite destacar palavras específicas;
- mostra a atenção com foco nos tokens mais relevantes para sua inspeção.

**Quando usar**  
Use quando quiser investigar o comportamento da atenção em partes específicas da sentença.
"""
    )

with st.expander("4. Mapas de Calor com Tokens Especiais (Plot 3)"):
    st.markdown(
        """
**Para que serve**  
Mostra mapas de calor mais detalhados da atenção, incluindo relações específicas entre pares de tokens.

**Entrada esperada**  
CSV com estas colunas:
- `sentence`
- `rule`
- `token_origem`
- `token_destino`
- `tokens_to_check`

**O que esta página faz**  
- permite escolher sentença e padrão;
- mostra pares de interesse;
- gera mapas de calor detalhados;
- facilita a inspeção de relações específicas entre tokens.

**Quando usar**  
Use quando precisar de uma análise mais minuciosa e estruturada da atenção.
"""
    )

with st.expander("5. Análise de Regras BERT (Regras)"):
    st.markdown(
        """
**Para que serve**  
Executa análises orientadas por padrões gramaticais e permite processar sentenças de forma mais sistemática.

**Entrada esperada**  
CSV com estas colunas:
- `sentence`
- `rule`
- `tokens_to_check`

**O que esta página faz**  
- permite escolher o modelo;
- permite selecionar uma sentença e um padrão;
- mostra os tokens a verificar;
- permite processar uma sentença isolada;
- permite processar várias sentenças e baixar os resultados em CSV.

**Quando usar**  
Use quando quiser gerar resultados tabulares mais amplos para análise posterior.
"""
    )

with st.expander("6. Treemap Interativo"):
    st.markdown(
        """
**Para que serve**  
Gera uma visualização hierárquica em treemap a partir de um CSV.

**Entrada esperada**  
- Um CSV com colunas adequadas para rótulo, pai e valor.

**O que esta página faz**  
- carrega o CSV;
- permite remover `[CLS]` e `[SEP]`;
- permite escolher as colunas do treemap;
- gera um gráfico interativo.

**Quando usar**  
Use para visualizar distribuição e hierarquia dos dados de forma resumida.
"""
    )

with st.expander("7. Heat Map"):
    st.markdown(
        """
**Para que serve**  
Gera um mapa de calor com a média da atenção por camada e cabeça.

**Entrada esperada**  
CSV com estas colunas:
- `Layer_Head`
- `Attention Value`
- `Layer`
- `Head`
- `sentence`
- `rule`

**O que esta página faz**  
- filtra por regra ou sentença;
- calcula a média de atenção;
- mostra um heatmap por camada e cabeça;
- ajuda a localizar regiões da arquitetura com maior ativação.

**Quando usar**  
Use quando quiser enxergar padrões globais de atenção no modelo.
"""
    )

with st.expander("Sobre as regras"):
    st.markdown("""
### Visão geral

Este sistema adota uma perspectiva inspirada em **Lucien Tesnière** e no esquema **Universal Dependencies (UD)**. Em vez de tratar a frase principalmente como combinação de sintagmas, a análise é feita como uma **rede de relações entre governantes e dependentes**. Assim, cada regra procura reconhecer um tipo recorrente de vínculo sintático entre palavras.

### Regras analisadas

**Verbo ditransitivo** *(verbo bitransitivo)*
Identifica um verbo que governa simultaneamente dois dependentes complementares, correspondendo a uma estrutura de valência mais ampla.

**Verbo monotransitivo direto** *(verbo transitivo direto)*
Identifica um verbo ligado diretamente a um único dependente que completa seu sentido.

**Complemento indireto do verbo** *(verbo transitivo indireto / objeto indireto)*
Identifica um dependente indireto subordinado a um verbo governante.

**Complemento oblíquo do verbo** *(complemento verbal preposicionado / adjunto adverbial, dependendo do caso)*
Identifica um dependente oblíquo ligado a um verbo. Em UD, essa categoria é mais ampla do que o “objeto indireto” da gramática tradicional.

**Dependente oracional** *(oração subordinada)*
Identifica uma oração inteira que depende estruturalmente de outro núcleo da sentença.

**Construção passiva** *(voz passiva)*
Identifica uma reorganização das dependências em torno de um predicado passivo, marcada por relações como `aux:pass` e `nsubj:pass`.

**Estrutura copulativa** *(verbo de ligação com predicativo do sujeito)*
Identifica construções em que a cópula depende de um núcleo predicativo semanticamente mais informativo, como em “está ameaçada”.

**Clítico pronominal** *(pronome reflexivo)*
Identifica pronomes clíticos, como “se”, “me” e “te”, ligados a um núcleo verbal.

**Modificador adverbial** *(adjunto adverbial)*
Identifica dependentes modificadores que acrescentam circunstância, intensidade ou modo a um governante.

### Observação final

As regras do sistema não devem ser entendidas apenas como rótulos tradicionais, mas como **padrões de dependência sintática** entre palavras. Isso torna a análise mais coerente com Tesnière e com a lógica estrutural de Universal Dependencies.
    """)

st.divider()

st.subheader("Qual página devo usar?")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
**Se você tem um `.conllu`**  
Comece em **Classificar Sentenças**.

**Se você já tem um CSV com sentenças e regras**  
Comece em **Plot 1** ou **Plot 2**.

**Se você quer focar em pares de tokens**  
Use **Plot 3**.
"""
    )

with col2:
    st.markdown(
        """
**Se você quer gerar saídas analíticas em lote**  
Use **Regras**.

**Se você quer visualização hierárquica**  
Use **Treemap**.

**Se você quer visão global por camada e cabeça**  
Use **Heat Map**.
"""
    )

st.info(
    """
Nem todas as páginas aceitam o mesmo formato de CSV.

Antes de fazer upload, verifique se o arquivo contém exatamente as colunas exigidas
pela página escolhida.
"""
)