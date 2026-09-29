"""Conversores dos datasets públicos sugeridos (PubMedQA e MedQuAD) para o formato interno.

Requer download manual (ou acesso à rede):
- PubMedQA: https://github.com/pubmedqa/pubmedqa  -> data/ori_pqal.json
- MedQuAD:  https://github.com/abachaa/MedQuAD   -> pastas com arquivos XML
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


def carregar_pubmedqa(path: Path, limite: int | None = None) -> list[dict]:
    dados = json.loads(path.read_text(encoding="utf-8"))
    exemplos = []
    for pmid, item in list(dados.items())[:limite]:
        contexto = " ".join(item.get("CONTEXTS", []))
        exemplos.append({
            "tipo": "pubmedqa",
            "fonte": f"PubMed PMID {pmid}",
            "pergunta": f"Contexto: {contexto}\nPergunta: {item['QUESTION']}\nResponda yes, no ou maybe e justifique.",
            "resposta": f"{item['final_decision']}. {item.get('LONG_ANSWER', '')} [PubMed PMID {pmid}]",
        })
    return exemplos


def carregar_medquad(root: Path, limite: int | None = None) -> list[dict]:
    exemplos = []
    for xml in sorted(root.rglob("*.xml")):
        doc = ET.parse(xml).getroot()
        url = doc.get("url", xml.name)
        for qa in doc.iter("QAPair"):
            q, a = qa.findtext("Question"), qa.findtext("Answer")
            if q and a:  # parte do MedQuAD tem respostas removidas por licença
                exemplos.append({"tipo": "medquad", "fonte": f"MedQuAD {url}", "pergunta": q.strip(),
                                 "resposta": f"{a.strip()} [MedQuAD {url}]"})
            if limite and len(exemplos) >= limite:
                return exemplos
    return exemplos
