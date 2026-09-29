"""Anonimização de textos clínicos (LGPD).

Substitui identificadores diretos por marcadores. Nomes são detectados por
contexto ("Paciente: ...", "Sr./Sra. ...", "Dr./Dra. ...") e, opcionalmente,
por uma lista de nomes conhecidos (ex.: vinda do cadastro de pacientes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("CNS", re.compile(r"\b[1-9]\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b")),
    ("RG", re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dXx]\b")),
    ("TELEFONE", re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b")),
    ("PRONTUARIO", re.compile(r"\b(?:prontu[aá]rio|registro|MRN)\s*(?:n[ºo°.]?\s*)?:?\s*\d{4,}\b", re.I)),
    ("DATA", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
]

_NAME = r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)*"
_NAME_CONTEXT = re.compile(
    rf"(?P<prefix>\b(?:[Pp]aciente|[Nn]ome|Sr\.?|Sra\.?|Dr\.?|Dra\.?|[Mm]ãe|[Pp]ai|[Aa]companhante)\s*:?\s+)(?P<name>{_NAME})"
)


@dataclass
class AnonymizationResult:
    text: str
    replacements: dict[str, int] = field(default_factory=dict)


def anonymize(text: str, known_names: list[str] | None = None) -> AnonymizationResult:
    counts: dict[str, int] = {}

    def sub(label: str, pattern: re.Pattern[str], value: str) -> str:
        new, n = pattern.subn(f"[{label}]", value)
        if n:
            counts[label] = counts.get(label, 0) + n
        return new

    out = text
    for name in sorted(known_names or [], key=len, reverse=True):
        out = sub("NOME", re.compile(re.escape(name), re.I), out)

    def repl_name(m: re.Match[str]) -> str:
        counts["NOME"] = counts.get("NOME", 0) + 1
        return m.group("prefix") + "[NOME]"

    out = _NAME_CONTEXT.sub(repl_name, out)
    for label, pattern in _PATTERNS:
        out = sub(label, pattern, out)
    return AnonymizationResult(out, counts)
