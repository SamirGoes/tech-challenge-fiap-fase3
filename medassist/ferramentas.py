"""Ferramentas (tools) do LangChain usadas pelo agente e pelo fluxo automatizado."""

from langchain_core.tools import tool

from medassist import clinica
from medassist.prontuario import buscar_paciente, resumo_clinico


def _paciente(paciente_id: int) -> dict:
    p = buscar_paciente(paciente_id)
    if p is None:
        raise ValueError(f"paciente {paciente_id} não encontrado")
    return p


@tool
def consultar_prontuario(paciente_id: int) -> str:
    """Resumo clínico do paciente (sem nome nem documentos) a partir da base de prontuários."""
    return resumo_clinico(_paciente(paciente_id))


@tool
def exames_pendentes(paciente_id: int) -> list[str]:
    """Lista os exames solicitados que ainda não têm resultado."""
    return clinica.exames_pendentes(_paciente(paciente_id))


@tool
def valores_criticos(paciente_id: int) -> list[str]:
    """Compara exames e sinais vitais com os limites do protocolo PROT-006 e lista os valores críticos."""
    return clinica.valores_criticos(_paciente(paciente_id))


FERRAMENTAS = [consultar_prontuario, exames_pendentes, valores_criticos]
