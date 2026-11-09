"""Assistente médico (LangChain):

    pergunta -> guardrail de entrada -> prontuário (SQL) -> RAG (Chroma) -> resposta -> guardrail de saída

Neste arquivo: guardrails, log de auditoria, base vetorial (RAG), resposta e a cadeia LangChain.
Por enquanto a resposta é montada só com o RAG (sem LLM); o fine-tuning fica na fase 2 (treino.py).
"""

import hashlib
import json
import math
import os
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableBranch, RunnableLambda

from medassist import dados
from medassist.dados import (AVISO, NAO_APLICA, RECUSA, SEM_PACIENTE, anonimizar, buscar_paciente, montar_prompt,
                             resumo_clinico)

# ---------------------------------------------------------------------------------------------------------
# Guardrails: limites de atuação do assistente
# ---------------------------------------------------------------------------------------------------------
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


def verificar_pergunta(pergunta):
    """Anonimiza a pergunta, bloqueia pedidos fora do escopo e marca pedidos de prescrição."""
    texto, pii = anonimizar(pergunta)
    if FORA_DO_ESCOPO.search(pergunta):
        return {"permitida": False, "motivo": "pedido fora do escopo ou tentativa de contornar as regras",
                "pede_prescricao": False, "pergunta": texto, "pii": pii}
    return {"permitida": True, "motivo": "", "pede_prescricao": bool(PEDIDO_PRESCRICAO.search(pergunta)),
            "pergunta": texto, "pii": pii}


def verificar_resposta(texto, pede_prescricao=False):
    """Retém prescrição direta, troca termos conclusivos, remove dados pessoais, exige fonte e anexa o aviso."""
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


# ---------------------------------------------------------------------------------------------------------
# Log de auditoria: uma linha por etapa, todas com o mesmo trace_id (console + logs/auditoria.jsonl)
# ---------------------------------------------------------------------------------------------------------
CONSOLE = True
_ordem, _ultimo = {}, {}


def novo_id():
    return uuid.uuid4().hex[:12]


def registrar(trace_id, etapa, **info):
    agora = time.perf_counter()
    ms = round((agora - _ultimo.get(trace_id, agora)) * 1000, 1)  # tempo desde a etapa anterior
    _ultimo[trace_id] = agora
    ordem = _ordem[trace_id] = _ordem.get(trace_id, 0) + 1
    linha = {"ts": datetime.now(timezone.utc).isoformat(), "trace_id": trace_id, "ordem": ordem, "etapa": etapa,
             "ms": ms, **info}
    dados.LOGS.mkdir(parents=True, exist_ok=True)
    with open(dados.LOGS / "auditoria.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(linha, ensure_ascii=False, default=str) + "\n")
    if CONSOLE:
        print(f"  [{trace_id}] {ordem}. {etapa}: {json.dumps(info, ensure_ascii=False, default=str)[:140]}")


def ler_log(trace_id=None):
    arquivo = dados.LOGS / "auditoria.jsonl"
    if not arquivo.exists():
        return []
    linhas = [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l]
    return [l for l in linhas if trace_id in (None, l["trace_id"])]


def linha_do_tempo(trace_id):
    saida = [f"trace_id {trace_id}"]
    for e in ler_log(trace_id):
        info = {k: v for k, v in e.items() if k not in ("ts", "trace_id", "ordem", "etapa", "ms")}
        saida.append(f"{e['ordem']:>2}. {e['etapa']:<16} {e['ms']:>7} ms  {json.dumps(info, ensure_ascii=False)[:150]}")
    return "\n".join(saida)


# ---------------------------------------------------------------------------------------------------------
# RAG: base vetorial Chroma, mutável (adicionar, atualizar e remover documentos sem mexer no código)
# ---------------------------------------------------------------------------------------------------------
IGNORAR = set("a o e de da do das dos em no na nos nas para com por um uma que qual quais como se ao os as ou "
              "mais segundo protocolo protocolos paciente conduta sugerida passos proximos aplicam".split())


def _tokens(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t[:6] for t in re.findall(r"[a-z0-9]+", texto) if t not in IGNORAR and len(t) > 1]


class EmbeddingsHash(Embeddings):
    """Embeddings simples (bag-of-words com hash): offline e reprodutível. Para embeddings de modelo, use
    MEDASSIST_EMBEDDINGS=hf (pip install langchain-huggingface sentence-transformers)."""

    def _vetor(self, texto, dim=1024):
        v = [0.0] * dim
        for t in set(_tokens(texto)):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % dim] += 1 if (h >> 64) % 2 else -1
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def embed_documents(self, textos):
        return [self._vetor(t) for t in textos]

    def embed_query(self, texto):
        return self._vetor(texto)


def abrir_indice(persistir=True, colecao=None):
    tipo = os.environ.get("MEDASSIST_EMBEDDINGS", "hash")
    if tipo == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    else:
        embeddings = EmbeddingsHash()
    extra = {"persist_directory": str(dados.PASTA / "chroma")} if persistir else {}
    return Chroma(collection_name=colecao or f"protocolos_{tipo}", embedding_function=embeddings,
                  collection_metadata={"hnsw:space": "cosine"}, **extra)


def documentos_protocolos():
    return [Document(page_content=f"{p['titulo']}. {secao.replace('_', ' ')}. {texto} {p['palavras']}",
                     metadata={"fonte": fonte, "documento": p["id"], "texto": texto})
            for fonte, p, secao, texto in dados.secoes()]


def documentos_de_arquivo(caminho):
    """Divide um .md ou .txt em trechos: por seção '## nome' ou, sem seções, por parágrafo."""
    caminho = Path(caminho)
    blocos = re.split(r"^## +(.+)$", caminho.read_text(encoding="utf-8-sig"), flags=re.M)
    if len(blocos) > 1:
        partes = [(blocos[i].strip(), blocos[i + 1].strip()) for i in range(1, len(blocos), 2)]
    else:
        partes = [(f"p{i}", p.strip()) for i, p in enumerate(re.split(r"\n\s*\n", blocos[0]), 1) if p.strip()]
    return [Document(page_content=f"{caminho.stem}. {secao}. {corpo}",
                     metadata={"fonte": f"DOC {caminho.stem} §{secao}", "documento": f"DOC {caminho.stem}", "texto": corpo})
            for secao, corpo in partes if corpo]


def adicionar(indice, docs):
    """Insere ou atualiza: o id é a fonte, então repetir a fonte substitui o trecho."""
    indice.add_documents(docs, ids=[d.metadata["fonte"] for d in docs])
    return len(docs)


def remover(indice, documento):
    ids = indice.get(where={"documento": documento})["ids"]
    if ids:
        indice.delete(ids=ids)
    return len(ids)


def reindexar(indice):
    ids = indice.get()["ids"]
    if ids:
        indice.delete(ids=ids)
    return adicionar(indice, documentos_protocolos())


def listar(indice):
    contagem = {}
    for m in indice.get()["metadatas"]:
        contagem[m["documento"]] = contagem.get(m["documento"], 0) + 1
    return dict(sorted(contagem.items()))


def buscar(indice, consulta, k=3, minimo=0.15):
    """Busca por similaridade de vetores (sem queries). Descarta trechos com score abaixo de `minimo`."""
    achados = [{"fonte": d.metadata["fonte"], "texto": d.metadata["texto"], "score": round(1 - dist, 3)}
               for d, dist in indice.similarity_search_with_score(consulta, k=k)]
    return [a for a in achados if a["score"] >= minimo]


# ---------------------------------------------------------------------------------------------------------
# Resposta (RAG puro: sem LLM). Qualquer LLM do LangChain pode entrar no lugar, ver treino.py
# ---------------------------------------------------------------------------------------------------------
class ModeloRAG(LLM):
    """Sem LLM: monta a resposta com os trechos de protocolo recuperados e os dados do prontuário."""

    @property
    def _llm_type(self):
        return "rag"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        pergunta = prompt.rsplit("### Pergunta", 1)[-1].strip()
        if PEDIDO_PRESCRICAO.search(pergunta):
            return RECUSA
        protocolos = prompt.split("### Protocolos", 1)[-1].split("### Paciente", 1)[0]
        trechos = re.findall(r"^\[([^\]]+)\] (.+)$", protocolos, re.M)
        if not trechos:
            return NAO_APLICA
        resposta = " ".join(f"{texto} [{fonte}]" for fonte, texto in trechos[:2])
        pendentes, pid = re.search(r"Pendentes: ([^.]+)\.", prompt), re.search(r"\[Prontuário #(\d+)\]", prompt)
        if pendentes and pid and pendentes.group(1) != "nenhum":
            resposta += f" Exames pendentes: {pendentes.group(1)} [Prontuário #{pid.group(1)}]."
        return resposta


# ---------------------------------------------------------------------------------------------------------
# Assistente: a cadeia LangChain
# ---------------------------------------------------------------------------------------------------------
class Assistente:
    def __init__(self, modelo=None, indice=None):
        self.llm = modelo or ModeloRAG()  # modelo: qualquer LLM do LangChain (fase 2); sem ele, RAG puro
        self.indice = indice or abrir_indice()
        if not self.indice.get(limit=1)["ids"]:
            reindexar(self.indice)
        etapa = lambda fn, nome: RunnableLambda(fn).with_config(run_name=nome)  # nome que aparece no LangSmith
        saida = etapa(self.saida, "saida")
        gerar = (etapa(self.prontuario, "prontuario") | etapa(self.rag, "rag_protocolos") | etapa(self.gerar, "resposta")
                 | etapa(self.validar, "guardrail_saida") | saida)
        bloquear = etapa(self.bloquear, "bloqueio") | saida
        self.cadeia = etapa(self.entrada, "entrada") | RunnableBranch((lambda e: not e["permitida"], bloquear), gerar)

    def entrada(self, e):
        r = verificar_pergunta(e["pergunta"])
        registrar(e["trace_id"], "entrada", usuario=e["usuario"], paciente_id=e["paciente_id"], pergunta=r["pergunta"],
                  pii_removida=r["pii"], permitida=r["permitida"], pede_prescricao=r["pede_prescricao"])
        return {**e, "pergunta": r["pergunta"], "permitida": r["permitida"], "pede_prescricao": r["pede_prescricao"],
                "motivo_bloqueio": r["motivo"]}

    def bloquear(self, e):
        return {**e, "texto": f"Pedido fora do escopo do assistente: {e['motivo_bloqueio']}.", "fontes": [],
                "violacoes": ["fora_do_escopo"], "revisao_humana": True}

    def prontuario(self, e):
        if e["paciente_id"] is None:
            registrar(e["trace_id"], "prontuario", consultado=False)
            return {**e, "paciente": SEM_PACIENTE, "paciente_ref": "-", "motivo": ""}
        p = buscar_paciente(e["paciente_id"])
        if p is None:
            raise ValueError(f"paciente {e['paciente_id']} não encontrado")
        registrar(e["trace_id"], "prontuario", consultado=True, paciente_id=p["id"],
                  exames_pendentes=dados.exames_pendentes(p))
        return {**e, "paciente": resumo_clinico(p), "paciente_ref": p["id"], "motivo": p["motivo"]}

    def rag(self, e):
        consulta = f"{e['pergunta']} {e['motivo']}".strip()
        trechos = buscar(self.indice, consulta)
        registrar(e["trace_id"], "rag", consulta=consulta, fontes=[(t["fonte"], t["score"]) for t in trechos])
        return {**e, "trechos": [(t["fonte"], t["texto"]) for t in trechos]}

    def gerar(self, e):
        prompt = montar_prompt(e["trechos"], e["paciente"], e["pergunta"], e["paciente_ref"])
        bruta = self.llm.invoke(prompt)
        registrar(e["trace_id"], "resposta", modelo=self.llm._llm_type, prompt=prompt, resposta_bruta=bruta)
        return {**e, "bruta": bruta}

    def validar(self, e):
        r = verificar_resposta(e["bruta"], e["pede_prescricao"])
        registrar(e["trace_id"], "guardrail_saida", violacoes=r["violacoes"], revisao_humana=r["revisao_humana"],
                  fontes=r["fontes"])
        return {**e, **r}

    def saida(self, e):
        resultado = {"trace_id": e["trace_id"], "texto": e["texto"], "fontes": e["fontes"],
                     "violacoes": e["violacoes"], "revisao_humana": e["revisao_humana"]}
        registrar(e["trace_id"], "saida", resposta_final=e["texto"], fontes=e["fontes"], violacoes=e["violacoes"],
                  revisao_humana=e["revisao_humana"])
        return resultado

    def perguntar(self, pergunta, paciente_id=None, usuario="medico", trace_id=None):
        trace_id = trace_id or novo_id()
        config = {"run_name": "medassist.perguntar", "metadata": {"trace_id": trace_id, "paciente_id": paciente_id}}
        return self.cadeia.invoke({"trace_id": trace_id, "pergunta": pergunta, "paciente_id": paciente_id,
                                   "usuario": usuario}, config)
