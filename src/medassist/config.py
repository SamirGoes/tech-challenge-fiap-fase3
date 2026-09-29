"""Configuração central. Todos os caminhos podem ser sobrescritos por variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _path("MEDASSIST_DATA_DIR", ROOT / "data"))
    db_path: Path = field(default_factory=lambda: _path("MEDASSIST_DB", ROOT / "data" / "hospital.db"))
    log_dir: Path = field(default_factory=lambda: _path("MEDASSIST_LOG_DIR", ROOT / "logs"))
    # "demo" (sem GPU, determinístico) ou "hf" (modelo base + adaptador LoRA treinado)
    llm_backend: str = field(default_factory=lambda: os.environ.get("MEDASSIST_LLM", "demo"))
    base_model: str = field(
        default_factory=lambda: os.environ.get("MEDASSIST_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    )
    adapter_dir: Path = field(
        default_factory=lambda: _path("MEDASSIST_ADAPTER", ROOT / "outputs" / "lora-medassist")
    )
    top_k_protocolos: int = 3

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def get_settings() -> Settings:
    return Settings()
