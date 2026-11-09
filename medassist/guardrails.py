"""Guardrails: limites de atuação do assistente, antes e depois do modelo.

Entrada: anonimiza a pergunta, bloqueia pedidos fora do escopo e marca pedidos de prescrição.
Saída: retém prescrição direta, troca termos conclusivos por linguagem de hipótese, remove dados pessoais,
exige fonte citada e anexa o aviso de validação médica.
"""

import re

from medassist.anonimizacao import anonimizar
from medassist.prompt import AVISO, RECUSA

PEDIDO_PRESCRICAO = re.compile(
    r"\b(prescrev\w*|receite|receitar|fa[cç]a (a )?receita|fazer (a )?receita|dose exata|qual a dose|que dose|"
    r"quantos (mg|ml)|posologia)\b", re.I)
FORA_DO_ESCOPO = re.compile(
    r"\b(ignore (as|suas|todas as) (instru\w+|regras)|esque[cç]a (as|suas) regras|atestado falso|burlar|"
    r"sem que o m[eé]dico saiba|dados de outro paciente)\b", re.I)
PRESCRICAO_DIRETA = re.compile(
    r"\b(prescrevo|prescreva-se|administre|tome|aplique)\b|"
    r"\b\d+(?:[.,]\d+)?\s?(mg|mcg|g|ml|UI)\b.{0,30}\b(a cada \d+ ?h|\d+/\d+ ?h|\dx ao dia)", re.I)
TERMO_CONCLUSIVO = re.compile(r"\bdiagn[oó]stic\w*", re.I)
FONTE = re.compile(r"\[((?:PROT-\d{3}|Prontuário|Política|LAUDO|RECEITA|DOC |PubMed|MedQuAD)[^\]]*)\]")


def verificar_entrada(pergunta: str) -> dict:
    texto, pii = anonimizar(pergunta)
    if FORA_DO_ESCOPO.search(pergunta):
        return {"permitida": False, "motivo": "pedido fora do escopo ou tentativa de contornar as regras",
                "pede_prescricao": False, "pergunta": texto, "pii": pii}
    return {"permitida": True, "motivo": "", "pede_prescricao": bool(PEDIDO_PRESCRICAO.search(pergunta)),
            "pergunta": texto, "pii": pii}


def verificar_resposta(texto: str, pede_prescricao: bool = False) -> dict:
    violacoes = []
    if PRESCRICAO_DIRETA.search(texto):
        violacoes.append("prescricao_direta")
        texto = "Resposta retida: continha linguagem de prescrição. Consulte o protocolo e o médico responsável."
    if TERMO_CONCLUSIVO.search(texto):
        violacoes.append("termo_conclusivo")
        texto = TERMO_CONCLUSIVO.sub("hipótese clínica", texto)
    texto, pii = anonimizar(texto)
    if pii:
        violacoes.append("dados_pessoais_na_saida")
    if pede_prescricao and "Não posso prescrever" not in texto:
        texto = f"{RECUSA}\n\n{texto}"
    fontes = sorted(set(FONTE.findall(texto)))
    if not fontes:
        violacoes.append("sem_fonte")
    return {"texto": f"{texto}\n\nAVISO: {AVISO}", "fontes": fontes, "violacoes": violacoes,
            "revisao_humana": pede_prescricao or bool(violacoes)}
