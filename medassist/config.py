"""Configuração central. Tudo pode ser trocado por variável de ambiente (lida a cada chamada)."""

import os
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    pasta_dados: Path
    pasta_conhecimento: Path
    pasta_logs: Path
    llm: str
    modelo_base: str
    adaptador: Path
    embeddings: str
    modelo_embeddings: str
    smtp_host: str
    smtp_porta: int
    smtp_usuario: str
    smtp_senha: str
    email_origem: str
    email_equipe_medica: str
    email_enfermagem: str

    @property
    def banco(self) -> Path:
        return self.pasta_dados / "hospital.db"

    @property
    def pasta_indice(self) -> Path:
        return self.pasta_dados / "chroma"


def config() -> Config:
    env = os.environ.get
    return Config(
        pasta_dados=Path(env("MEDASSIST_DADOS", RAIZ / "dados")),
        pasta_conhecimento=Path(env("MEDASSIST_CONHECIMENTO", RAIZ / "conhecimento")),
        pasta_logs=Path(env("MEDASSIST_LOGS", RAIZ / "logs")),
        llm=env("MEDASSIST_LLM", "demo"),
        modelo_base=env("MEDASSIST_MODELO", "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"),
        adaptador=Path(env("MEDASSIST_ADAPTADOR", RAIZ / "saida" / "lora-medassist")),
        embeddings=env("MEDASSIST_EMBEDDINGS", "hash"),
        modelo_embeddings=env("MEDASSIST_MODELO_EMBEDDINGS", "intfloat/multilingual-e5-small"),
        smtp_host=env("MEDASSIST_SMTP_HOST", ""),
        smtp_porta=int(env("MEDASSIST_SMTP_PORTA", "587")),
        smtp_usuario=env("MEDASSIST_SMTP_USUARIO", ""),
        smtp_senha=env("MEDASSIST_SMTP_SENHA", ""),
        email_origem=env("MEDASSIST_EMAIL_ORIGEM", "medassist@hospital.exemplo"),
        email_equipe_medica=env("MEDASSIST_EMAIL_EQUIPE_MEDICA", "equipe.medica@hospital.exemplo"),
        email_enfermagem=env("MEDASSIST_EMAIL_ENFERMAGEM", "enfermagem@hospital.exemplo"),
    )


def ativar_langsmith() -> bool:
    """Liga o rastreamento no LangSmith (opcional) se houver chave de API no ambiente."""
    chave = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if chave and os.environ.get("MEDASSIST_LANGSMITH", "1") != "0":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "medassist-fase3")
        return True
    return False
