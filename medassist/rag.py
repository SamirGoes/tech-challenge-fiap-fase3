"""RAG sobre base vetorial (Chroma). A base é mutável: dá para adicionar, atualizar e remover documentos.

Os protocolos de `conhecimento/` são indexados por seção. Documentos novos entram com `adicionar_arquivo`,
sem alterar código. A busca é semântica por similaridade de vetores (nenhuma query SQL ou filtro de texto).
Embeddings: "hash" (padrão, offline e determinístico) ou "hf" (modelo multilíngue do Hugging Face).
"""

import hashlib
import math
import re
import unicodedata
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from medassist.config import config
from medassist.protocolos import secoes

IGNORAR = set("a o e de da do das dos em no na nos nas para com por um uma que qual quais como se ao os as ou "
              "mais segundo protocolo protocolos paciente conduta sugerida passos proximos aplicam".split())


def _tokens(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t[:6] for t in re.findall(r"[a-z0-9]+", texto) if t not in IGNORAR and len(t) > 1]


class EmbeddingsHash(Embeddings):
    """Bag-of-words com hashing (radicais de 6 letras), normalizado. Leve, offline e reprodutível."""

    def __init__(self, dimensao: int = 1024) -> None:
        self.dimensao = dimensao

    def _vetor(self, texto: str) -> list[float]:
        v = [0.0] * self.dimensao
        contagem = {}
        for t in _tokens(texto):
            contagem[t] = contagem.get(t, 0) + 1
        for t, c in contagem.items():
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dimensao] += (1 + math.log(c)) * (1 if (h >> 64) % 2 else -1)
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        return [self._vetor(t) for t in textos]

    def embed_query(self, texto: str) -> list[float]:
        return self._vetor(texto)


def criar_embeddings(tipo: str | None = None) -> Embeddings:
    c = config()
    tipo = tipo or c.embeddings
    if tipo == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface sentence-transformers
        return HuggingFaceEmbeddings(model_name=c.modelo_embeddings)
    return EmbeddingsHash()


def documentos_protocolos() -> list[Document]:
    docs = []
    for fonte, p, secao, texto in secoes():
        docs.append(Document(
            page_content=f"{p['titulo']}. {secao.replace('_', ' ')}. {texto} {p['palavras']}",
            metadata={"fonte": fonte, "documento": p["id"], "secao": secao, "titulo": p["titulo"], "texto": texto}))
    return docs


def documentos_de_arquivo(caminho) -> list[Document]:
    """Divide um .md/.txt em trechos: por seção '## nome' ou, sem seções, por parágrafo."""
    caminho = Path(caminho)
    nome = f"DOC {caminho.stem}"
    texto = caminho.read_text(encoding="utf-8-sig")  # aceita arquivos com BOM (comum no Windows)
    blocos = re.split(r"^## +(.+)$", texto, flags=re.M)
    if len(blocos) > 1:
        partes = [(blocos[i].strip(), blocos[i + 1].strip()) for i in range(1, len(blocos), 2)]
    else:
        partes = [(f"p{i}", p.strip()) for i, p in enumerate(re.split(r"\n\s*\n", texto), 1) if p.strip()]
    return [Document(page_content=f"{caminho.stem}. {secao}. {corpo}",
                     metadata={"fonte": f"{nome} §{secao}", "documento": nome, "secao": secao,
                               "titulo": caminho.stem, "texto": corpo})
            for secao, corpo in partes if corpo]


class IndiceProtocolos:
    def __init__(self, persistir: bool = True, colecao: str | None = None, embeddings: Embeddings | None = None) -> None:
        c = config()
        self.embeddings = embeddings or criar_embeddings()
        nome = colecao or f"protocolos_{c.embeddings}"
        argumentos = {"persist_directory": str(c.pasta_indice)} if persistir else {}
        self.base = Chroma(collection_name=nome, embedding_function=self.embeddings,
                           collection_metadata={"hnsw:space": "cosine"}, **argumentos)

    def total(self) -> int:
        return self.base._collection.count()

    def adicionar(self, docs: list[Document]) -> int:
        """Insere ou atualiza (o id é a fonte, então repetir a fonte substitui o trecho)."""
        self.base.add_documents(docs, ids=[d.metadata["fonte"] for d in docs])
        return len(docs)

    def adicionar_arquivo(self, caminho) -> int:
        return self.adicionar(documentos_de_arquivo(caminho))

    def remover(self, documento: str) -> int:
        """Remove todos os trechos de um documento (ex.: 'PROT-001' ou 'DOC nome')."""
        ids = self.base.get(where={"documento": documento})["ids"]
        if ids:
            self.base.delete(ids=ids)
        return len(ids)

    def reindexar(self) -> int:
        ids = self.base.get()["ids"]
        if ids:
            self.base.delete(ids=ids)
        return self.adicionar(documentos_protocolos())

    def garantir(self) -> None:
        if self.total() == 0:
            self.adicionar(documentos_protocolos())

    def listar(self) -> dict[str, int]:
        contagem = {}
        for m in self.base.get()["metadatas"]:
            contagem[m["documento"]] = contagem.get(m["documento"], 0) + 1
        return dict(sorted(contagem.items()))

    def buscar(self, consulta: str, k: int = 3) -> list[dict]:
        resultados = self.base.similarity_search_with_score(consulta, k=k)  # distância cosseno
        return [{"fonte": d.metadata["fonte"], "texto": d.metadata["texto"], "score": round(1 - dist, 3)}
                for d, dist in resultados]


class RetrieverProtocolos(BaseRetriever):
    """Retriever do LangChain sobre o índice vetorial; guarda fonte e score nos metadados."""

    indice: object
    k: int = 3
    score_minimo: float = 0.15  # abaixo disso o trecho é ruído e não entra no prompt

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [Document(page_content=r["texto"], metadata={"fonte": r["fonte"], "score": r["score"]})
                for r in self.indice.buscar(query, self.k) if r["score"] >= self.score_minimo]


def indice_em_memoria() -> IndiceProtocolos:
    """Índice efêmero e isolado (testes e demonstrações), já populado com os protocolos."""
    indice = IndiceProtocolos(persistir=False, colecao=f"mem_{uuid.uuid4().hex[:8]}")
    indice.garantir()
    return indice
