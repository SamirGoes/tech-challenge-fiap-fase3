"""Modelos de linguagem expostos como `LLM` do LangChain.

- `demo`: extrativo e determinístico (sem GPU). Serve para testes, para demonstrar o fluxo e como baseline.
- `treinado`: modelo base (4 bits) + adaptador LoRA do fine-tuning. `base` usa o mesmo modelo sem o adaptador.
"""

import json
import re
from typing import Any

from langchain_core.language_models.llms import LLM

from medassist.config import config
from medassist.guardrails import PEDIDO_PRESCRICAO
from medassist.prompt import RECUSA, SEM_PROTOCOLO, SYSTEM_PROMPT


class ModeloDemo(LLM):
    @property
    def _llm_type(self) -> str:
        return "demo"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
        pergunta = prompt.rsplit("### Pergunta", 1)[-1].strip()
        if PEDIDO_PRESCRICAO.search(pergunta):
            return RECUSA
        protocolos = prompt.split("### Protocolos", 1)[-1].split("### Paciente", 1)[0]
        trechos = re.findall(r"^\[([^\]]+)\] (.+)$", protocolos, re.M)
        if not trechos or SEM_PROTOCOLO in protocolos:
            return "Nenhum protocolo interno aplicável foi encontrado. [Política MedAssist]"
        resposta = " ".join(f"{texto} [{fonte}]" for fonte, texto in trechos[:2])
        pendentes = re.search(r"Pendentes: ([^.]+)\.", prompt)
        pid = re.search(r"\[Prontuário #(\d+)\]", prompt)
        if pendentes and pid and pendentes.group(1) != "nenhum":
            resposta += f" Exames pendentes: {pendentes.group(1)} [Prontuário #{pid.group(1)}]."
        return resposta


class ModeloTreinado(LLM):
    usar_adaptador: bool = True
    max_tokens: int = 384
    _modelo: Any = None
    _tokenizer: Any = None

    @property
    def _llm_type(self) -> str:
        return "qwen-lora" if self.usar_adaptador else "qwen-base"

    def _carregar(self) -> None:
        if self._modelo is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        c = config()
        base = c.modelo_base
        if self.usar_adaptador:
            base = json.loads((c.adaptador / "adapter_config.json").read_text())["base_model_name_or_path"]
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        opcoes = {"device_map": "auto", "torch_dtype": torch.float16}
        if torch.cuda.is_available() and "bnb-4bit" not in base:
            from transformers import BitsAndBytesConfig
            opcoes["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.float16)
        modelo = AutoModelForCausalLM.from_pretrained(base, **opcoes)
        if self.usar_adaptador:
            from peft import PeftModel
            modelo = PeftModel.from_pretrained(modelo, str(c.adaptador))
        self._modelo = modelo.eval()

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
        import torch

        self._carregar()
        mensagens = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        entrada = self._tokenizer.apply_chat_template(mensagens, add_generation_prompt=True, return_tensors="pt")
        entrada = entrada.to(self._modelo.device)
        with torch.no_grad():
            saida = self._modelo.generate(entrada, max_new_tokens=self.max_tokens, do_sample=False,
                                          pad_token_id=self._tokenizer.eos_token_id)
        return self._tokenizer.decode(saida[0, entrada.shape[1]:], skip_special_tokens=True).strip()


_CACHE: dict[str, LLM] = {}


def criar_modelo(tipo: str | None = None) -> LLM:
    """tipo: demo | base | treinado. Os modelos pesados são reaproveitados entre chamadas."""
    tipo = tipo or config().llm
    if tipo not in ("demo", "base", "treinado"):
        raise ValueError(f"modelo desconhecido: {tipo}")
    if tipo == "demo":
        return ModeloDemo()
    if tipo not in _CACHE:
        _CACHE[tipo] = ModeloTreinado(usar_adaptador=tipo == "treinado")
    return _CACHE[tipo]
