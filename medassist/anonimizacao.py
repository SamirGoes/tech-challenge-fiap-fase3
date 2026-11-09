"""Anonimização de textos: troca dados pessoais por marcadores como [NOME] e [CPF]."""

import re

REGRAS_PII = [
    ("CPF", r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    ("TELEFONE", r"\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b"),
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("DATA", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    ("PRONTUARIO", r"\bprontu[aá]rio\s*:?\s*\d{4,}\b"),
]
MAIUSCULA = "A-ZÁÉÍÓÚÂÊÔÃÕÇ"
NOME_COM_CONTEXTO = (rf"(\b(?:Paciente|Sr\.|Sra\.|Dr\.|Dra\.)\s*:?\s+)[{MAIUSCULA}][\wáéíóúâêôãõç]+"
                     rf"(?:\s+[{MAIUSCULA}][\wáéíóúâêôãõç]+)*")


def anonimizar(texto: str, nomes=()) -> tuple[str, dict]:
    """Devolve (texto_anonimizado, {tipo: quantidade}). `nomes` é uma lista de nomes conhecidos."""
    contagem = {}
    for nome in nomes:
        texto, n = re.subn(re.escape(nome), "[NOME]", texto, flags=re.I)
        contagem["NOME"] = contagem.get("NOME", 0) + n
    texto, n = re.subn(NOME_COM_CONTEXTO, r"\1[NOME]", texto)
    contagem["NOME"] = contagem.get("NOME", 0) + n
    for rotulo, regra in REGRAS_PII:
        texto, n = re.subn(regra, f"[{rotulo}]", texto, flags=re.I)
        contagem[rotulo] = contagem.get(rotulo, 0) + n
    return texto, {k: v for k, v in contagem.items() if v}
