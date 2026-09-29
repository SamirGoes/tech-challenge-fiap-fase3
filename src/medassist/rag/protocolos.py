"""Recuperação das seções de protocolo relevantes (BM25), exposta como Retriever do LangChain.

Cada Document carrega a fonte citável em `metadata["fonte"]`, o que permite
mostrar ao médico de onde veio cada parte da resposta.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from medassist.data.protocolos import iter_secoes

_STOP = set("a o e de da do das dos em no na nos nas para com por um uma que qual quais como se ao os as "
            "é ou mais segundo protocolo protocolos paciente quais conduta sugerida inicial passos proximos aplicam".split())


def _tok(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", texto) if t not in _STOP and len(t) > 1]


def documentos_protocolos() -> list[Document]:
    docs = []
    for fonte, p, secao, texto in iter_secoes():
        docs.append(Document(
            page_content=f"{p['titulo']} - {secao.replace('_', ' ')}: {texto}",
            metadata={"fonte": fonte, "protocolo": p["id"], "secao": secao,
                      "palavras_chave": " ".join(p["palavras_chave"])},
        ))
    return docs


class ProtocoloRetriever(BaseRetriever):
    k: int = 3
    _docs: list[Document] = PrivateAttr(default_factory=list)
    _tf: list[Counter] = PrivateAttr(default_factory=list)
    _idf: dict[str, float] = PrivateAttr(default_factory=dict)
    _avg: float = PrivateAttr(default=1.0)

    def model_post_init(self, __context) -> None:
        self._docs = documentos_protocolos()
        # palavras-chave do protocolo contam em dobro: descrevem os quadros clínicos a que ele se aplica
        self._tf = [Counter(_tok(d.page_content + (" " + d.metadata["palavras_chave"]) * 2)) for d in self._docs]
        n = len(self._docs)
        df = Counter(t for tf in self._tf for t in tf)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._avg = sum(sum(tf.values()) for tf in self._tf) / n

    def pontuar(self, query: str) -> list[tuple[float, Document]]:
        q = _tok(query)
        k1, b = 1.5, 0.75
        res = []
        for tf, doc in zip(self._tf, self._docs):
            dl = sum(tf.values())
            s = sum(self._idf.get(t, 0) * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self._avg))
                    for t in q if t in tf)
            if s > 0:
                res.append((s, doc))
        return sorted(res, key=lambda x: -x[0])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        out = []
        for score, doc in self.pontuar(query)[: self.k]:
            out.append(Document(page_content=doc.page_content, metadata={**doc.metadata, "score": round(score, 3)}))
        return out
