import pytest

from medassist import dados, prontuario
from medassist.agente import AgenteMedico
from medassist.auditoria import Auditoria
from medassist.llm import ModeloDemo
from medassist.rag import indice_em_memoria


@pytest.fixture(autouse=True)
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDASSIST_DADOS", str(tmp_path / "dados"))
    monkeypatch.setenv("MEDASSIST_LOGS", str(tmp_path / "logs"))
    monkeypatch.delenv("MEDASSIST_SMTP_HOST", raising=False)
    dados.preparar()
    prontuario.criar_banco()


@pytest.fixture
def auditoria():
    return Auditoria()


@pytest.fixture
def agente(auditoria):
    return AgenteMedico(llm=ModeloDemo(), indice=indice_em_memoria(), auditoria=auditoria)
