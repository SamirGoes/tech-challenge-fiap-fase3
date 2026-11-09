"""Leitura dos protocolos internos (arquivos .md em conhecimento/).

Formato: "# Título", linhas "id:" e "palavras:", e uma seção "## nome" por trecho citável.
Cada seção vira uma fonte, por exemplo "PROT-001 §primeira_hora".
"""

from medassist.config import config


def ler_protocolo(caminho) -> dict:
    protocolo = {"id": caminho.stem, "titulo": caminho.stem, "palavras": "", "secoes": {}}
    secao = None
    for linha in caminho.read_text(encoding="utf-8-sig").splitlines():
        linha = linha.strip()
        if linha.startswith("## "):
            secao = linha[3:].strip()
            protocolo["secoes"][secao] = ""
        elif linha.startswith("# "):
            protocolo["titulo"] = linha[2:].strip()
        elif secao is None and linha.startswith("id:"):
            protocolo["id"] = linha[3:].strip()
        elif secao is None and linha.startswith("palavras:"):
            protocolo["palavras"] = linha[9:].strip()
        elif secao and linha:
            protocolo["secoes"][secao] = (protocolo["secoes"][secao] + " " + linha).strip()
    return protocolo


def carregar_protocolos(pasta=None) -> dict[str, dict]:
    pasta = pasta or config().pasta_conhecimento
    protocolos = [ler_protocolo(c) for c in sorted(pasta.glob("PROT-*.md"))]
    return {p["id"]: p for p in protocolos}


def secoes(pasta=None):
    """Gera (fonte, protocolo, nome_da_secao, texto)."""
    for pid, p in carregar_protocolos(pasta).items():
        for nome, texto in p["secoes"].items():
            yield f"{pid} §{nome}", p, nome, texto
