"""Pipeline LangChain do assistente: prontuário + protocolos + LLM customizada + guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from medassist.audit.logger import AuditLogger, new_trace_id
from medassist.db.prontuario import ProntuarioRepository, resumo_clinico
from medassist.llm.factory import get_llm
from medassist.rag.protocolos import ProtocoloRetriever
from medassist.safety.guardrails import verificar_entrada, verificar_saida

PROMPT = PromptTemplate.from_template(
    "Use apenas as informações abaixo. Cite a fonte entre colchetes após cada afirmação. "
    "Não prescreva: apresente sugestões para validação do médico responsável.\n\n"
    "### Protocolos internos\n{protocolos}\n\n"
    "### Dados atualizados do paciente [Prontuário #{paciente_id}]\n{paciente}\n\n"
    "### Pergunta\n{pergunta}"
)


def _formatar(docs) -> str:
    return "\n".join(f"[{d.metadata['fonte']}] {d.page_content}" for d in docs) or "(nenhum protocolo relevante)"


@dataclass
class RespostaAssistente:
    trace_id: str
    resposta: str
    fontes: list[str]
    requer_revisao_humana: bool
    violacoes: list[str] = field(default_factory=list)
    bloqueado: bool = False


class AssistenteMedico:
    def __init__(self, llm=None, repo: ProntuarioRepository | None = None,
                 retriever: ProtocoloRetriever | None = None, audit: AuditLogger | None = None) -> None:
        self.llm = llm or get_llm()
        self.repo = repo or ProntuarioRepository()
        self.retriever = retriever or ProtocoloRetriever()
        self.audit = audit or AuditLogger()
        # entrada {pergunta, paciente, paciente_id} -> + docs -> + protocolos -> + resposta
        self.chain = (
            RunnablePassthrough.assign(docs=RunnableLambda(self._recuperar))
            | RunnablePassthrough.assign(protocolos=RunnableLambda(lambda e: _formatar(e["docs"])))
            | RunnablePassthrough.assign(resposta=PROMPT | self.llm | StrOutputParser())
        )

    def _recuperar(self, entrada: dict):
        # busca pela pergunta + motivo de internação; sinais vitais e exames entram só no prompt
        return self.retriever.invoke(f"{entrada['pergunta']} {entrada.get('motivo', '')}")

    def perguntar(self, pergunta: str, paciente_id: int | None = None, usuario: str = "medico",
                  trace_id: str | None = None) -> RespostaAssistente:
        trace_id = trace_id or new_trace_id()
        self.audit.log(trace_id, "pergunta_recebida", usuario=usuario, paciente_id=paciente_id, pergunta=pergunta)

        entrada_ok = verificar_entrada(pergunta)
        if not entrada_ok.permitido:
            self.audit.log(trace_id, "pergunta_bloqueada", motivo=entrada_ok.motivo)
            return RespostaAssistente(trace_id, f"Pedido não atendido: {entrada_ok.motivo}.", [], True,
                                      ["fora_escopo"], bloqueado=True)

        paciente_txt, pid, motivo = "(nenhum paciente selecionado)", "-", ""
        if paciente_id is not None:
            pac = self.repo.paciente(paciente_id)
            if pac is None:
                raise ValueError(f"paciente {paciente_id} não encontrado")
            paciente_txt, pid, motivo = resumo_clinico(pac), paciente_id, pac["motivo_internacao"]
            self.audit.log(trace_id, "prontuario_consultado", paciente_id=paciente_id,
                           exames_pendentes=[e["nome"] for e in pac["exames"] if e["status"] == "pendente"])

        entrada = {"pergunta": pergunta, "paciente": paciente_txt, "paciente_id": pid, "motivo": motivo}
        resultado = self.chain.invoke(entrada)
        bruta, docs = resultado["resposta"], resultado["docs"]
        self.audit.log(trace_id, "contexto_recuperado",
                       fontes=[{"fonte": d.metadata["fonte"], "score": d.metadata.get("score")} for d in docs])
        self.audit.log(trace_id, "resposta_llm", llm=self.llm._llm_type, resposta_bruta=bruta)

        saida = verificar_saida(bruta, entrada_ok.pedido_prescricao)
        self.audit.log(trace_id, "validacao_seguranca", aprovado=saida.aprovado, violacoes=saida.violacoes,
                       requer_revisao_humana=saida.requer_revisao_humana, fontes=saida.fontes)
        return RespostaAssistente(trace_id, saida.texto, saida.fontes, saida.requer_revisao_humana, saida.violacoes)
