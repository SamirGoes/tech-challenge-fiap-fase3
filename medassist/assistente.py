import os
import re
import unicodedata

from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from medassist import seguranca
from medassist.dados import RECUSA, SYSTEM_PROMPT, secoes
from medassist.prontuario import buscar_paciente, resumo_clinico

MODELO_BASE = os.environ.get("MEDASSIST_MODELO", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTADOR = os.environ.get("MEDASSIST_ADAPTADOR", "saida/lora-medassist")
IGNORAR = set("a o e de da do das dos em no na para com por um uma que qual quais como se os as ou "
              "protocolo paciente conduta segundo".split())

PROMPT = PromptTemplate.from_template(
    "Use apenas as informações abaixo e cite a fonte entre colchetes. Não prescreva.\n\n"
    "### Protocolos\n{protocolos}\n\n### Paciente [Prontuário #{paciente_id}]\n{paciente}\n\n### Pergunta\n{pergunta}"
)


def palavras(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return {p for p in re.findall(r"[a-z0-9]+", texto) if p not in IGNORAR and len(p) > 1}


def buscar_protocolos(consulta, k=2):
    alvo = palavras(consulta)
    pontuados = []
    for fonte, p, secao, texto in secoes():
        pontos = len(alvo & palavras(f"{secao} {texto}")) + 2 * len(alvo & palavras(f"{p['titulo']} {p['palavras']}"))
        if pontos:
            pontuados.append((pontos, fonte, texto))
    pontuados.sort(key=lambda x: -x[0])
    return [{"fonte": fonte, "texto": texto, "pontos": pontos} for pontos, fonte, texto in pontuados[:k]]


class ModeloDemo(LLM):
    @property
    def _llm_type(self):
        return "demo"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        pergunta = prompt.rsplit("### Pergunta", 1)[-1]
        if seguranca.verificar_pergunta(pergunta)["pede_prescricao"]:
            return RECUSA
        trechos = re.findall(r"^\[([^\]]+)\] (.+)$", prompt, re.M)
        if not trechos:
            trechos = [(d["fonte"], d["texto"]) for d in buscar_protocolos(pergunta, k=1)]
        return " ".join(f"{texto} [{fonte}]" for fonte, texto in trechos) or "Nenhum protocolo encontrado. [Política MedAssist]"


class ModeloTreinado(LLM):
    usar_adaptador: bool = True
    _modelo: object = None
    _tokenizer: object = None

    @property
    def _llm_type(self):
        return "qwen-lora"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._modelo is None:
            self._tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
            base = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float16, device_map="auto")
            self._modelo = PeftModel.from_pretrained(base, ADAPTADOR) if self.usar_adaptador else base
        mensagens = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        entrada = self._tokenizer.apply_chat_template(mensagens, add_generation_prompt=True, return_tensors="pt")
        entrada = entrada.to(self._modelo.device)
        with torch.no_grad():
            saida = self._modelo.generate(entrada, max_new_tokens=384, do_sample=False)
        return self._tokenizer.decode(saida[0, entrada.shape[1]:], skip_special_tokens=True).strip()


def criar_modelo(tipo=None):
    tipo = tipo or os.environ.get("MEDASSIST_LLM", "demo")
    if tipo in ("treinado", "base"):
        return ModeloTreinado(usar_adaptador=tipo == "treinado")
    return ModeloDemo()


def criar_cadeia(modelo):
    return (
        RunnablePassthrough.assign(docs=RunnableLambda(lambda e: buscar_protocolos(f"{e['pergunta']} {e['motivo']}")))
        | RunnablePassthrough.assign(protocolos=RunnableLambda(
            lambda e: "\n".join(f"[{d['fonte']}] {d['texto']}" for d in e["docs"]) or "(nenhum)"))
        | RunnablePassthrough.assign(resposta=PROMPT | modelo | StrOutputParser())
    )


def perguntar(pergunta, paciente_id=None, modelo=None, trace_id=None, usuario="medico"):
    trace_id = trace_id or seguranca.novo_id()
    seguranca.registrar(trace_id, "pergunta", usuario=usuario, paciente_id=paciente_id, pergunta=pergunta)

    checagem = seguranca.verificar_pergunta(pergunta)
    if not checagem["permitida"]:
        seguranca.registrar(trace_id, "bloqueada")
        return {"trace_id": trace_id, "texto": "Pedido fora do escopo do assistente.", "fontes": [],
                "violacoes": ["fora_do_escopo"], "revisao_humana": True}

    entrada = {"pergunta": pergunta, "paciente": "(nenhum)", "paciente_id": "-", "motivo": ""}
    if paciente_id is not None:
        p = buscar_paciente(paciente_id)
        if p is None:
            raise ValueError(f"Paciente {paciente_id} não encontrado")
        entrada.update(paciente=resumo_clinico(p), paciente_id=paciente_id, motivo=p["motivo"])
        seguranca.registrar(trace_id, "prontuario_consultado", paciente_id=paciente_id)

    modelo = modelo or criar_modelo()
    resultado = criar_cadeia(modelo).invoke(entrada)
    seguranca.registrar(trace_id, "protocolos_usados", fontes=[(d["fonte"], d["pontos"]) for d in resultado["docs"]])
    seguranca.registrar(trace_id, "resposta_modelo", modelo=modelo._llm_type, texto=resultado["resposta"])

    final = seguranca.verificar_resposta(resultado["resposta"], checagem["pede_prescricao"])
    seguranca.registrar(trace_id, "validacao", violacoes=final["violacoes"], revisao_humana=final["revisao_humana"])
    return {"trace_id": trace_id, **final}
