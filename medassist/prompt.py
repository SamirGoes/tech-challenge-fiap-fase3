"""Prompt único usado no treino, na avaliação e no assistente.

Treinar com o mesmo formato usado em produção (protocolos + paciente + pergunta) ensina o modelo
a usar o contexto recuperado pelo RAG e a citar as fontes.
"""

from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT = (
    "Você é o MedAssist, assistente de apoio à decisão clínica do hospital. "
    "Responda com base nos protocolos internos e nos dados do paciente, cite a fonte entre colchetes "
    "e nunca prescreva: toda conduta ou medicamento exige validação do médico responsável."
)

RECUSA = ("Não posso prescrever nem definir doses. Posso resumir o que o protocolo recomenda e os dados do paciente "
          "para apoiar a sua decisão; a prescrição deve ser feita pelo médico responsável. [Política MedAssist]")

AVISO = ("Apoio à decisão clínica: não substitui a avaliação do médico e não é prescrição. "
         "Consulte o médico responsável antes de qualquer conduta.")

SEM_PROTOCOLO = "(nenhum protocolo relevante)"
SEM_PACIENTE = "(nenhum paciente selecionado)"

PROMPT = PromptTemplate.from_template(
    "Use apenas as informações abaixo e cite a fonte entre colchetes. Não prescreva.\n\n"
    "### Protocolos\n{protocolos}\n\n### Paciente [Prontuário #{paciente_id}]\n{paciente}\n\n### Pergunta\n{pergunta}"
)


def formatar_protocolos(trechos: list[tuple[str, str]]) -> str:
    """trechos: lista de (fonte, texto)."""
    return "\n".join(f"[{fonte}] {texto}" for fonte, texto in trechos) or SEM_PROTOCOLO


def montar_prompt(protocolos: str, paciente: str, pergunta: str, paciente_id="-") -> str:
    return PROMPT.format(protocolos=protocolos, paciente=paciente, paciente_id=paciente_id, pergunta=pergunta)
