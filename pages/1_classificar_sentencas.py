import io
import streamlit as st
import pandas as pd
from conllu import parse_incr

# st.set_page_config(
#     page_title="Analisador de Padrões Gramaticais — Universal Dependencies",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

st.title("Analisador de Padrões Gramaticais — Universal Dependencies")

# ============================================================
# UTILITÁRIOS
# ============================================================
def get_token_by_id(tokens, token_id):
    """
    Recupera o token pelo ID real.
    Evita assumir que tokens[token_id - 1] corresponde sempre ao head.
    """
    if not isinstance(token_id, int) or token_id <= 0:
        return None
    return next((tok for tok in tokens if tok.get("id") == token_id), None)


def valid_token(tok):
    return isinstance(tok.get("id"), int)


def sentence_text_from_tokens(tokens):
    """
    Fallback para sentenças sem metadata['text'].
    """
    return " ".join(tok.get("form", "") for tok in tokens if valid_token(tok)).strip()


def normalize_tokens(sentence):
    """
    Mantém apenas tokens básicos com campos necessários.
    """
    valid_tokens = [tok for tok in sentence if valid_token(tok)]
    return [
        {
            "id": tok.get("id"),
            "form": tok.get("form"),
            "lemma": tok.get("lemma"),
            "upos": tok.get("upos"),
            "deprel": tok.get("deprel"),
            "head": tok.get("head"),
        }
        for tok in valid_tokens
    ]


def append_classification(results, sentence_text, rule, governor, dependent):
    results.append(
        {
            "sentence": sentence_text,
            "rule": rule,
            "governante": governor,
            "dependente": dependent,
        }
    )


# ============================================================
# REGRAS DE CLASSIFICAÇÃO
# Perspectiva de dependências: governante -> dependente
# ============================================================
grammar_rules = {
    "Verbo ditransitivo": {
        "conditions": lambda tokens: any(
            tok["upos"] == "VERB"
            and any(ch["head"] == tok["id"] and ch["deprel"] == "obj" for ch in tokens)
            and any(ch["head"] == tok["id"] and ch["deprel"] == "iobj" for ch in tokens)
            for tok in tokens
        )
    },
    "Verbo monotransitivo direto": {
        "conditions": lambda tokens: any(
            tok["deprel"] == "obj"
            and (head := get_token_by_id(tokens, tok["head"])) is not None
            and head["upos"] == "VERB"
            and not any(ch["head"] == head["id"] and ch["deprel"] == "iobj" for ch in tokens)
            for tok in tokens
        )
    },
    "Complemento indireto do verbo": {
        "conditions": lambda tokens: any(
            tok["deprel"] == "iobj"
            and (head := get_token_by_id(tokens, tok["head"])) is not None
            and head["upos"] == "VERB"
            for tok in tokens
        )
    },
    "Complemento oblíquo do verbo": {
        "conditions": lambda tokens: any(
            tok["deprel"] == "obl"
            and (head := get_token_by_id(tokens, tok["head"])) is not None
            and head["upos"] == "VERB"
            for tok in tokens
        )
    },
    "Dependente oracional": {
        "conditions": lambda tokens: any(
            tok["deprel"] in {"csubj", "ccomp", "xcomp", "advcl", "acl:relcl"}
            and get_token_by_id(tokens, tok["head"]) is not None
            for tok in tokens
        )
    },
    "Construção passiva": {
        "conditions": lambda tokens: any(
            tok["deprel"] in {"aux:pass", "nsubj:pass"}
            for tok in tokens
        )
    },
    "Estrutura copulativa": {
        "conditions": lambda tokens: any(
            tok["deprel"] == "cop"
            and get_token_by_id(tokens, tok["head"]) is not None
            for tok in tokens
        )
    },
    "Clítico pronominal": {
        "conditions": lambda tokens: any(
            str(tok["deprel"]).startswith("expl")
            and str(tok["form"]).lower() in {"se", "me", "te", "nos", "vos"}
            and get_token_by_id(tokens, tok["head"]) is not None
            for tok in tokens
        )
    },
    "Modificador adverbial": {
        "conditions": lambda tokens: any(
            tok["deprel"] == "advmod"
            and get_token_by_id(tokens, tok["head"]) is not None
            for tok in tokens
        )
    },
}

# ============================================================
# REGRAS DE EXTRAÇÃO
# Cada padrão retorna pares governante -> dependente
# ============================================================
grammatical_patterns = {
    "Verbo ditransitivo": {
        "conditions": lambda tokens: [
            (
                tok["id"],
                tok["form"],
                f'{obj["form"]} | {iobj["form"]}',
                f'{obj["id"]} | {iobj["id"]}',
            )
            for tok in tokens
            if tok["upos"] == "VERB"
            for obj in [next((ch for ch in tokens if ch["head"] == tok["id"] and ch["deprel"] == "obj"), None)]
            for iobj in [next((ch for ch in tokens if ch["head"] == tok["id"] and ch["deprel"] == "iobj"), None)]
            if obj is not None and iobj is not None
        ]
    },
    "Verbo monotransitivo direto": {
        "conditions": lambda tokens: [
            (head["id"], head["form"], tok["id"], tok["form"])
            for tok in tokens
            if tok["deprel"] == "obj"
            for head in [get_token_by_id(tokens, tok["head"])]
            if head is not None
            and head["upos"] == "VERB"
            and not any(ch["head"] == head["id"] and ch["deprel"] == "iobj" for ch in tokens)
        ]
    },
    "Complemento indireto do verbo": {
        "conditions": lambda tokens: [
            (head["id"], head["form"], tok["id"], tok["form"])
            for tok in tokens
            if tok["deprel"] == "iobj"
            for head in [get_token_by_id(tokens, tok["head"])]
            if head is not None and head["upos"] == "VERB"
        ]
    },
    "Complemento oblíquo do verbo": {
        "conditions": lambda tokens: [
            (head["id"], head["form"], tok["id"], tok["form"])
            for tok in tokens
            if tok["deprel"] == "obl"
            for head in [get_token_by_id(tokens, tok["head"])]
            if head is not None and head["upos"] == "VERB"
        ]
    },
    "Dependente oracional": {
        "conditions": lambda tokens: [
            (head["id"], head["form"], tok["id"], tok["form"])
            for tok in tokens
            if tok["deprel"] in {"csubj", "ccomp", "xcomp", "advcl", "acl:relcl"}
            for head in [get_token_by_id(tokens, tok["head"])]
            if head is not None
        ]
    },
    "Construção passiva": {
        "conditions": lambda tokens: [
            (head["id"], head["form"], tok["id"], tok["form"])
            for tok in tokens
            if tok["deprel"] in {"aux:pass", "nsubj:pass"}
            for head in [get_token_by_id(tokens, tok["head"])]
            if head is not None
        ]
    },
    "Estrutura copulativa": {
        "conditions": lambda tokens: [
            (head["id"], head["form"], tok["id"], tok["form"])
            for tok in tokens
            if tok["deprel"] == "cop"
            for head in [get_token_by_id(tokens, tok["head"])]
            if head is not None
        ]
    },
    "Clítico pronominal": {
        "conditions": lambda tokens: [
            (head["id"], head["form"], tok["id"], tok["form"])
            for tok in tokens
            if str(tok["deprel"]).startswith("expl")
            and str(tok["form"]).lower() in {"se", "me", "te", "nos", "vos"}
            for head in [get_token_by_id(tokens, tok["head"])]
            if head is not None
        ]
    },
    "Modificador adverbial": {
        "conditions": lambda tokens: [
            (head["id"], head["form"], tok["id"], tok["form"])
            for tok in tokens
            if tok["deprel"] == "advmod"
            for head in [get_token_by_id(tokens, tok["head"])]
            if head is not None
        ]
    },
}

# Ordem de prioridade na classificação
RULE_PRIORITY = [
    "Verbo ditransitivo",
    "Verbo monotransitivo direto",
    "Complemento indireto do verbo",
    "Complemento oblíquo do verbo",
    "Dependente oracional",
    "Construção passiva",
    "Estrutura copulativa",
    "Clítico pronominal",
    "Modificador adverbial",
]

# ============================================================
# UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Faça upload de um arquivo .conllu", type=["conllu"])

if uploaded_file:
    try:
        raw_text = uploaded_file.getvalue().decode("utf-8")
        sentences = list(parse_incr(io.StringIO(raw_text)))
        st.success("Arquivo carregado com sucesso!")

        # ============================================================
        # DEBUG / INSPEÇÃO
        # ============================================================
        with st.expander("Diagnóstico do corpus", expanded=False):
            st.write(f"Total de sentenças lidas: {len(sentences)}")

            all_tokens = []
            for sent in sentences:
                toks = normalize_tokens(sent)
                all_tokens.extend(toks)

            if all_tokens:
                df_debug = pd.DataFrame(all_tokens)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Frequência de DEPREL**")
                    st.dataframe(
                        df_debug["deprel"].value_counts(dropna=False).reset_index().rename(
                            columns={"index": "deprel", "deprel": "freq"}
                        ),
                        use_container_width=True,
                    )

                with col2:
                    st.markdown("**Frequência de UPOS**")
                    st.dataframe(
                        df_debug["upos"].value_counts(dropna=False).reset_index().rename(
                            columns={"index": "upos", "upos": "freq"}
                        ),
                        use_container_width=True,
                    )

                st.markdown("**Amostra de tokens**")
                st.dataframe(df_debug.head(30), use_container_width=True)

        # ============================================================
        # CLASSIFICAÇÃO
        # ============================================================
        sentence_rules = []

        for idx, sentence in enumerate(sentences):
            valid_tokens = normalize_tokens(sentence)
            text = sentence.metadata.get("text", "").strip() if sentence.metadata else ""
            if not text:
                text = sentence_text_from_tokens(valid_tokens)

            matched_rule = None

            for rule in RULE_PRIORITY:
                cond = grammar_rules[rule]
                if cond["conditions"](valid_tokens):
                    matched_rule = rule
                    break

            if matched_rule is None:
                append_classification(sentence_rules, text, "Não classificada", "-", "-")
                continue

            # Extrai o primeiro par representativo da regra escolhida
            matches = grammatical_patterns[matched_rule]["conditions"](valid_tokens)

            if matches:
                origem_id, origem_form, destino_id, destino_form = matches[0]
                append_classification(
                    sentence_rules,
                    text,
                    matched_rule,
                    origem_form,
                    destino_form,
                )
            else:
                # fallback defensivo: classificou, mas não extraiu
                append_classification(
                    sentence_rules,
                    text,
                    matched_rule,
                    "(não identificado)",
                    "(não identificado)",
                )

        df_classified = pd.DataFrame(sentence_rules)

        with st.expander("Distribuição das regras", expanded=False):
            if not df_classified.empty:
                st.dataframe(
                    df_classified["rule"].value_counts(dropna=False).reset_index().rename(
                        columns={"index": "rule", "rule": "freq"}
                    ),
                    use_container_width=True,
                )

        df_classified_show = df_classified[df_classified["rule"] != "Não classificada"].copy()

        st.subheader("Classificação Geral com Governante e Dependente")
        if not df_classified_show.empty:
            st.dataframe(df_classified_show.head(50), use_container_width=True)
        else:
            st.warning("Nenhuma sentença foi classificada pelas regras atuais.")

        # ============================================================
        # ESTRUTURAÇÃO DAS SENTENÇAS CLASSIFICADAS
        # ============================================================
        sentence_to_rule = dict(zip(df_classified_show["sentence"], df_classified_show["rule"]))
        sentences_structured = {}

        for idx, sentence in enumerate(sentences):
            valid_tokens = normalize_tokens(sentence)
            text = sentence.metadata.get("text", "").strip() if sentence.metadata else ""
            if not text:
                text = sentence_text_from_tokens(valid_tokens)

            sent_id = sentence.metadata.get("sent_id", f"sent_{idx+1}") if sentence.metadata else f"sent_{idx+1}"

            if text in sentence_to_rule:
                sentences_structured[sent_id] = {
                    "Sentence": text,
                    "Tokens": valid_tokens,
                }

        # ============================================================
        # EXTRAÇÃO DOS PADRÕES
        # ============================================================
        resultados = []

        for sent_id, sent_data in sentences_structured.items():
            tokens = sent_data["Tokens"]
            sentence_text = sent_data["Sentence"]

            regra = sentence_to_rule.get(sentence_text)
            if regra in grammatical_patterns:
                matches = grammatical_patterns[regra]["conditions"](tokens)
            else:
                matches = []

            for origem_id, origem_form, destino_id, destino_form in matches:
                resultados.append(
                    {
                        "Sentence ID": sent_id,
                        "Sentence": sentence_text,
                        "Pattern": regra,
                        "Origin Token": origem_form,
                        "Origin ID": origem_id,
                        "Destination Token": destino_form,
                        "Destination ID": destino_id,
                    }
                )

        df_resultado = pd.DataFrame(resultados)

        # ============================================================
        # SAÍDA
        # ============================================================
        if not df_resultado.empty:
            st.subheader("Padrões Identificados")
            st.dataframe(df_resultado.head(50), use_container_width=True)

            df_tokens_export = df_resultado[
                ["Sentence", "Pattern", "Origin Token", "Destination Token"]
            ].copy()

            df_tokens_export["Tokens Concatenados"] = df_tokens_export.apply(
                lambda row: [f'"{row["Origin Token"]}"', f'"{row["Destination Token"]}"'],
                axis=1,
            )

            df_tokens_export = df_tokens_export.rename(
                columns={
                    "Sentence": "sentence",
                    "Pattern": "rule",
                    "Origin Token": "token_origem",
                    "Destination Token": "token_destino",
                    "Tokens Concatenados": "tokens_to_check",
                }
            )

            csv_buffer = io.StringIO()
            df_tokens_export.to_csv(csv_buffer, index=False, encoding="utf-8")

            st.download_button(
                label="Baixar checar_tokens.csv",
                data=csv_buffer.getvalue(),
                file_name="checar_tokens.csv",
                mime="text/csv",
            )
        else:
            st.warning("Nenhum padrão identificado. Nenhum arquivo será gerado.")

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

else:
    st.info("Faça upload de um arquivo .conllu para começar.")