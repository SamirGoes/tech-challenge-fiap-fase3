import pytest

from medassist.audit.logger import AuditLogger
from medassist.chains.assistant import AssistenteMedico
from medassist.db.prontuario import ProntuarioRepository, inicializar
from medassist.llm.factory import DemoLLM


@pytest.fixture
def audit(tmp_path):
    return AuditLogger(tmp_path / "audit.jsonl")


@pytest.fixture
def repo(tmp_path):
    return ProntuarioRepository(inicializar(tmp_path / "hospital.db"))


@pytest.fixture
def assistente(repo, audit):
    return AssistenteMedico(llm=DemoLLM(), repo=repo, audit=audit)
