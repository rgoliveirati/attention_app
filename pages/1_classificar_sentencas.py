import streamlit as st
import pandas as pd
from conllu import parse_incr
import io

# st.set_page_config(
#     page_title="Analisador de Padrões Gramaticais — Universal Dependencies",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

st.title("Analisador de Padrões Gramaticais — Universal Dependencies")

# =======================
# Utilitário
# =======================
def get_token_by_id(tokens, token_id):
    """
    Recupera o token cujo campo 'id' é igual a token_id.
    Evita assumir que tokens[token_id - 1] sempre corresponde ao head,
    o que pode falhar em casos com IDs não triviais.
    """
    if not isinstance(token_id, int) or token_id <= 0:
        return None
    return next((tok for tok in tokens if tok.get("id") == token_id), None)


# =======================
# Regras de Classificação Geral
# =======================
grammar_rules = {
    "Verbo bitransitivo": {
        "conditions": lambda tokens: any(
            tok["upos"] == "VERB"
            and any(child["deprel"] == "obj" and child["head"] == tok["id"] for child in tokens)
            and any(child["deprel"] == "iobj" and child["head"] == tok["id"] for child in tokens)
            for tok in tokens
        )
    },
    "Verbo transitivo direto": {
        "conditions": lambda tokens: any(
            tok["deprel"] == "obj"
            and get_token_by_id(tokens, tok["head"]) is not None
            and get_token_by_id(tokens, tok["head"])["upos"] == "VERB"
            for tok in tokens
        )
    },
    "Verbo transitivo indireto": {
        "conditions": lambda tokens: any(
            tok["deprel"] in {"iobj", "obl"}
            and get_token_by_id(tokens, tok["head"]) is not None
            and get_token_by_id(tokens, tok["head"])["upos"] == "VERB"
            for tok in tokens
        )
    },
    "Oração subordinada": {
        "conditions": lambda tokens: any(
            tok["deprel"] in {"ccomp", "advcl", "xcomp", "acl:relcl", "mark"}
            for tok in tokens
        )
    },
    "Voz passiva": {
        "conditions": lambda tokens: any(
            tok["deprel"] in {"aux:pass", "nsubj:pass"}
            for tok in tokens
        )
    },
    "Verbo com predicativo do sujeito": {
        "conditions": lambda tokens: any(
            tok["deprel"] == "cop"
            and get_token_by_id(tokens, tok["head"]) is not None
            and get_token_by_id(tokens, tok["head"])["upos"] in {"VERB", "ADJ", "NOUN", "PROPN"}
            for tok in tokens
        )
    },
    "Pronome reflexivo": {
        "conditions": lambda tokens: any(
            str(tok["deprel"]).startswith("expl")
            and tok["form"].lower() in {"se", "me", "te", "nos", "vos"}
            for tok in tokens
        )
    },
    "Adjunto adverbial": {
        "conditions": lambda tokens: any(
            tok["deprel"] == "advmod"
            and get_token_by_id(tokens, tok["head"]) is not None
            for tok in tokens
        )
    }
}

# =======================
# Regras de Extração de Padrões
# =======================
grammatical_patterns = {
    "Verbo bitransitivo": {
        "conditions": lambda tokens: [
            (tok["id"], tok["form"], child["id"], child["form"])
            for tok in tokens
            if tok["upos"] == "VERB"
            for child in tokens
            if child["head"] == tok["id"] and child["deprel"] in {"obj", "iobj"}
        ]
    },
    "Verbo transitivo direto": {
        "conditions": lambda tokens: [
            (
                get_token_by_id(tokens, tok["head"])["id"],
                get_token_by_id(tokens, tok["head"])["form"],
                tok["id"],
                tok["form"]
            )
            for tok in tokens
            if tok["deprel"] == "obj"
            and get_token_by_id(tokens, tok["head"]) is not None
            and get_token_by_id(tokens, tok["head"])["upos"] == "VERB"
        ]
    },
    "Verbo transitivo indireto": {
        "conditions": lambda tokens: [
            (
                get_token_by_id(tokens, tok["head"])["id"],
                get_token_by_id(tokens, tok["head"])["form"],
                tok["id"],
                tok["form"]
            )
            for tok in tokens
            if tok["deprel"] in {"iobj", "obl"}
            and get_token_by_id(tokens, tok["head"]) is not None
            and get_token_by_id(tokens, tok["head"])["upos"] == "VERB"
        ]
    },
    "Oração subordinada": {
        "conditions": lambda tokens: [
            (
                get_token_by_id(tokens, tok["head"])["id"],
                get_token_by_id(tokens, tok["head"])["form"],
                tok["id"],
                tok["form"]
            )
            for tok in tokens
            if tok["deprel"] in {"csubj", "ccomp", "advcl", "xcomp", "acl:relcl", "mark"}
            and get_token_by_id(tokens, tok["head"]) is not None
        ]
    },
    "Voz passiva": {
        "conditions": lambda tokens: [
            (
                get_token_by_id(tokens, tok["head"])["id"],
                get_token_by_id(tokens, tok["head"])["form"],
                tok["id"],
                tok["form"]
            )
            for tok in tokens
            if tok["deprel"] in {"aux:pass", "nsubj:pass"}
            and get_token_by_id(tokens, tok["head"]) is not None
        ]
    },
    "Verbo com predicativo do sujeito": {
        "conditions": lambda tokens: [
            (
                get_token_by_id(tokens, tok["head"])["id"],
                get_token_by_id(tokens, tok["head"])["form"],
                tok["id"],
                tok["form"]
            )
            for tok in tokens
            if tok["deprel"] == "cop"
            and get_token_by_id(tokens, tok["head"]) is not None
        ]
    },
    "Pronome reflexivo": {
        "conditions": lambda tokens: [
            (
                get_token_by_id(tokens, tok["head"])["id"],
                get_token_by_id(tokens, tok["head"])["form"],
                tok["id"],
                tok["form"]
            )
            for tok in tokens
            if str(tok["deprel"]).startswith("expl")
            and get_token_by_id(tokens, tok["head"]) is not None
        ]
    },
    "Adjunto adverbial": {
        "conditions": lambda tokens: [
            (
                get_token_by_id(tokens, tok["head"])["id"],
                get_token_by_id(tokens, tok["head"])["form"],
                tok["id"],
                tok["form"]
            )
            for tok in tokens
            if tok["deprel"] == "advmod"
            and get_token_by_id(tokens, tok["head"]) is not None
        ]
    }
}

# =======================
# Upload do Arquivo
# =======================
uploaded_file = st.file_uploader("Faça upload de um arquivo .conllu", type=["conllu"])

if uploaded_file:
    st.success("Arquivo carregado com sucesso!")

    sentences = list(parse_incr(io.StringIO(uploaded_file.getvalue().decode("utf-8"))))

    # =======================
    # Classificação Geral das Sentenças com Governante e Dependente
    # =======================
    sentence_rules = []

    for sentence in sentences:
        text = sentence.metadata.get("text", "N/A")
        matched = False

        # manter apenas tokens com id inteiro
        valid_tokens = [tok for tok in sentence if isinstance(tok.get("id"), int)]

        for rule, cond in grammar_rules.items():
            if cond["conditions"](valid_tokens):
                matched = True

                for tok in valid_tokens:
                    if rule == "Verbo bitransitivo" and tok["upos"] == "VERB":
                        obj = next(
                            (child for child in valid_tokens if child["deprel"] == "obj" and child["head"] == tok["id"]),
                            None
                        )
                        iobj = next(
                            (child for child in valid_tokens if child["deprel"] == "iobj" and child["head"] == tok["id"]),
                            None
                        )
                        if obj and iobj:
                            sentence_rules.append({
                                "sentence": text,
                                "rule": rule,
                                "governante": tok["form"],
                                "dependente": f'{obj["form"]}, {iobj["form"]}'
                            })
                            break

                    elif rule == "Verbo transitivo direto":
                        obj = next(
                            (
                                t for t in valid_tokens
                                if t["deprel"] == "obj"
                                and get_token_by_id(valid_tokens, t["head"]) is not None
                                and get_token_by_id(valid_tokens, t["head"])["upos"] == "VERB"
                            ),
                            None
                        )
                        if obj:
                            head = get_token_by_id(valid_tokens, obj["head"])
                            sentence_rules.append({
                                "sentence": text,
                                "rule": rule,
                                "governante": head["form"],
                                "dependente": obj["form"]
                            })
                            break

                    elif rule == "Verbo transitivo indireto":
                        iobj = next(
                            (
                                t for t in valid_tokens
                                if t["deprel"] in {"iobj", "obl"}
                                and get_token_by_id(valid_tokens, t["head"]) is not None
                                and get_token_by_id(valid_tokens, t["head"])["upos"] == "VERB"
                            ),
                            None
                        )
                        if iobj:
                            head = get_token_by_id(valid_tokens, iobj["head"])
                            sentence_rules.append({
                                "sentence": text,
                                "rule": rule,
                                "governante": head["form"],
                                "dependente": iobj["form"]
                            })
                            break

                    elif rule == "Oração subordinada":
                        sub = next(
                            (
                                t for t in valid_tokens
                                if t["deprel"] in {"ccomp", "advcl", "xcomp", "acl:relcl", "mark"}
                                and get_token_by_id(valid_tokens, t["head"]) is not None
                            ),
                            None
                        )
                        if sub:
                            head = get_token_by_id(valid_tokens, sub["head"])
                            sentence_rules.append({
                                "sentence": text,
                                "rule": rule,
                                "governante": head["form"],
                                "dependente": sub["form"]
                            })
                            break

                    elif rule == "Voz passiva":
                        passive = next(
                            (
                                t for t in valid_tokens
                                if t["deprel"] in {"aux:pass", "nsubj:pass"}
                                and get_token_by_id(valid_tokens, t["head"]) is not None
                            ),
                            None
                        )
                        if passive:
                            head = get_token_by_id(valid_tokens, passive["head"])
                            sentence_rules.append({
                                "sentence": text,
                                "rule": rule,
                                "governante": head["form"],
                                "dependente": passive["form"]
                            })
                            break

                    elif rule == "Verbo com predicativo do sujeito":
                        cop = next(
                            (
                                t for t in valid_tokens
                                if t["deprel"] == "cop"
                                and get_token_by_id(valid_tokens, t["head"]) is not None
                            ),
                            None
                        )
                        if cop:
                            head = get_token_by_id(valid_tokens, cop["head"])
                            sentence_rules.append({
                                "sentence": text,
                                "rule": rule,
                                "governante": head["form"],
                                "dependente": cop["form"]
                            })
                            break

                    elif rule == "Pronome reflexivo":
                        expl = next(
                            (
                                t for t in valid_tokens
                                if str(t["deprel"]).startswith("expl")
                                and t["form"].lower() in {"se", "me", "te", "nos", "vos"}
                                and get_token_by_id(valid_tokens, t["head"]) is not None
                            ),
                            None
                        )
                        if expl:
                            head = get_token_by_id(valid_tokens, expl["head"])
                            sentence_rules.append({
                                "sentence": text,
                                "rule": rule,
                                "governante": head["form"],
                                "dependente": expl["form"]
                            })
                            break

                    elif rule == "Adjunto adverbial":
                        adv = next(
                            (
                                t for t in valid_tokens
                                if t["deprel"] == "advmod"
                                and get_token_by_id(valid_tokens, t["head"]) is not None
                            ),
                            None
                        )
                        if adv:
                            head = get_token_by_id(valid_tokens, adv["head"])
                            sentence_rules.append({
                                "sentence": text,
                                "rule": rule,
                                "governante": head["form"],
                                "dependente": adv["form"]
                            })
                            break

                # encerra a busca após registrar a primeira ocorrência daquela regra
                if sentence_rules and sentence_rules[-1]["sentence"] == text and sentence_rules[-1]["rule"] == rule:
                    break

        if not matched:
            sentence_rules.append({
                "sentence": text,
                "rule": "Não classificada",
                "governante": "-",
                "dependente": "-"
            })

    df_classified = pd.DataFrame(sentence_rules)
    df_classified = df_classified[df_classified["rule"] != "Não classificada"]

    st.subheader("Classificação Geral com Governante e Dependente")
    st.dataframe(df_classified.head(10))

    # =======================
    # Estruturar Sentenças para Padrões
    # =======================
    sentence_to_rule = dict(zip(df_classified["sentence"], df_classified["rule"]))

    sentences_structured = {}

    for idx, sentence in enumerate(sentences):
        text = sentence.metadata.get("text", "N/A")
        sent_id = sentence.metadata.get("sent_id", f"sent_{idx+1}")

        if text in sentence_to_rule:
            valid_tokens = [tok for tok in sentence if isinstance(tok.get("id"), int)]
            sentences_structured[sent_id] = {
                "Sentence": text,
                "Tokens": [
                    {
                        "id": tok["id"],
                        "form": tok["form"],
                        "upos": tok["upos"],
                        "deprel": tok["deprel"],
                        "head": tok["head"]
                    }
                    for tok in valid_tokens
                ]
            }

    # =======================
    # Extração de Padrões Governante–Dependente
    # =======================
    resultados = []

    for sent_id, sent_data in sentences_structured.items():
        tokens = sent_data["Tokens"]
        sentence_text = sent_data["Sentence"]

        for regra, config in grammatical_patterns.items():
            matches = config["conditions"](tokens)
            for origem_id, origem_form, destino_id, destino_form in matches:
                resultados.append({
                    "Sentence ID": sent_id,
                    "Sentence": sentence_text,
                    "Pattern": regra,
                    "Origin Token": origem_form,
                    "Origin ID": origem_id,
                    "Destination Token": destino_form,
                    "Destination ID": destino_id
                })

    df_resultado = pd.DataFrame(resultados)

    if not df_resultado.empty:
        st.subheader("Padrões Identificados")
        st.dataframe(df_resultado.head(10))

        df_tokens_export = df_resultado[["Sentence", "Pattern", "Origin Token", "Destination Token"]].copy()
        df_tokens_export["Tokens Concatenados"] = df_tokens_export.apply(
            lambda row: [f'"{row["Origin Token"]}"', f'"{row["Destination Token"]}"'],
            axis=1
        )
        df_tokens_export = df_tokens_export.rename(columns={
            "Sentence": "sentence",
            "Pattern": "rule",
            "Origin Token": "token_origem",
            "Destination Token": "token_destino",
            "Tokens Concatenados": "tokens_to_check"
        })

        csv_buffer = io.StringIO()
        df_tokens_export.to_csv(csv_buffer, index=False, encoding="utf-8")

        st.download_button(
            label="Baixar checar_tokens.csv",
            data=csv_buffer.getvalue(),
            file_name="checar_tokens.csv",
            mime="text/csv"
        )
    else:
        st.warning("Nenhum padrão identificado. Nenhum arquivo será gerado.")