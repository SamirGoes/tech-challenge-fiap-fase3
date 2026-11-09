"""Log de auditoria de ponta a ponta: cada requisição gera eventos ordenados com o mesmo `trace_id`.

Cada evento é uma linha JSON em logs/auditoria.jsonl e também aparece no console. Um evento por etapa:
entrada -> prontuario -> rag -> llm -> guardrail_saida -> saida (e, no fluxo, revisão e alertas).
"""

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from medassist.config import config

console = logging.getLogger("medassist")
if not console.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s medassist | %(message)s", "%H:%M:%S"))
    console.addHandler(_h)
    console.setLevel(logging.INFO)


def novo_id() -> str:
    return uuid.uuid4().hex[:12]


def _curto(dados: dict, limite: int = 200) -> str:
    texto = json.dumps(dados, ensure_ascii=False, default=str)
    return texto if len(texto) <= limite else texto[:limite] + "..."


class Auditoria:
    def __init__(self, caminho=None) -> None:
        self.caminho = caminho or config().pasta_logs / "auditoria.jsonl"
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._ordem: dict[str, int] = {}
        self._trava = threading.Lock()

    def registrar(self, trace_id: str, etapa: str, **dados) -> dict:
        with self._trava:  # etapas em paralelo (LangChain) escrevem do mesmo processo
            self._ordem[trace_id] = self._ordem.get(trace_id, 0) + 1
            linha = {"ts": datetime.now(timezone.utc).isoformat(), "trace_id": trace_id,
                     "ordem": self._ordem[trace_id], "etapa": etapa, **dados}
            with open(self.caminho, "a", encoding="utf-8") as f:
                f.write(json.dumps(linha, ensure_ascii=False, default=str) + "\n")
        console.info("[%s] #%s %s %s", trace_id, linha["ordem"], etapa, _curto(dados))
        return linha

    @contextmanager
    def etapa(self, trace_id: str, nome: str, **inicio):
        """Mede a duração da etapa e registra status ok/erro. Use `registro.update(...)` para anexar dados."""
        registro = dict(inicio)
        t0 = time.perf_counter()
        try:
            yield registro
        except Exception as erro:
            self.registrar(trace_id, nome, status="erro", erro=repr(erro),
                           duracao_ms=round((time.perf_counter() - t0) * 1000, 1), **registro)
            raise
        self.registrar(trace_id, nome, status="ok", duracao_ms=round((time.perf_counter() - t0) * 1000, 1), **registro)

    def ler(self, trace_id: str | None = None) -> list[dict]:
        if not self.caminho.exists():
            return []
        linhas = [json.loads(l) for l in self.caminho.read_text(encoding="utf-8").splitlines() if l]
        return [l for l in linhas if trace_id in (None, l["trace_id"])]

    def linha_do_tempo(self, trace_id: str) -> str:
        eventos = self.ler(trace_id)
        if not eventos:
            return f"Nenhum evento para o trace_id {trace_id}."
        saida = [f"trace_id {trace_id}"]
        for e in eventos:
            extra = {k: v for k, v in e.items() if k not in ("ts", "trace_id", "ordem", "etapa", "status", "duracao_ms")}
            saida.append(f"{e['ordem']:>2}. {e['etapa']:<18} {e.get('status', 'ok'):<5} "
                         f"{str(e.get('duracao_ms', '')):>8} ms  {_curto(extra, 150)}")
        return "\n".join(saida)
