"""Backends de LLM expostos como `LLM` do LangChain.

- `hf`: modelo base do Hugging Face + adaptador LoRA treinado em `finetune/train_lora.py`.
- `demo`: backend extrativo e determinístico (sem GPU) que responde com os trechos de
  protocolo mais relevantes. Serve para testes, para a demo em CPU e como baseline de avaliação.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.llms import LLM

from medassist.config import get_settings
from medassist.data.synthetic import SYSTEM_PROMPT
from medassist.safety.guardrails import verificar_entrada

_BLOCO_FONTE = re.compile(r"^\[(?P<fonte>[^\]]+)\]\s*(?P<texto>.+)$", re.M)


class DemoLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "medassist-demo"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
        pergunta = prompt.rsplit("### Pergunta", 1)[-1].strip()
        if verificar_entrada(pergunta).pedido_prescricao:
            return ("Não posso prescrever nem definir doses diretamente. Posso resumir o que o protocolo interno "
                    "recomenda e os dados relevantes do paciente para apoiar a sua decisão; a prescrição deve ser "
                    "feita e assinada pelo médico responsável. [Política de uso do MedAssist]")
        blocos = [(m["fonte"], m["texto"]) for m in _BLOCO_FONTE.finditer(prompt)]
        if not blocos:  # sem contexto no prompt: recupera por conta própria (baseline de avaliação)
            from medassist.rag.protocolos import ProtocoloRetriever

            docs = ProtocoloRetriever(k=1).invoke(pergunta)
            blocos = [(d.metadata["fonte"], d.page_content.split(": ", 1)[-1]) for d in docs]
        if not blocos:
            return "Não encontrei protocolo interno aplicável a esta pergunta. [Política de uso do MedAssist]"
        fonte, texto = blocos[0]
        resposta = f"{texto} [{fonte}]"
        if len(blocos) > 1:
            resposta += f"\nVer também: {blocos[1][1]} [{blocos[1][0]}]"
        return resposta


class HFChatLLM(LLM):
    base_model: str
    adapter_dir: str | None = None
    max_new_tokens: int = 384
    _model: Any = None
    _tok: Any = None

    @property
    def _llm_type(self) -> str:
        return "medassist-hf-lora"

    def _carregar(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.adapter_dir or self.base_model)
        model = AutoModelForCausalLM.from_pretrained(self.base_model, torch_dtype=torch.float16, device_map="auto")
        if self.adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_dir)
        self._model = model.eval()

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
        import torch

        self._carregar()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        entrada = self._tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        entrada = entrada.to(self._model.device)
        with torch.no_grad():
            saida = self._model.generate(entrada, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         repetition_penalty=1.1, pad_token_id=self._tok.eos_token_id)
        return self._tok.decode(saida[0, entrada.shape[1]:], skip_special_tokens=True).strip()


def get_llm(backend: str | None = None, use_adapter: bool = True) -> LLM:
    s = get_settings()
    backend = backend or s.llm_backend
    if backend == "demo":
        return DemoLLM()
    if backend == "hf":
        adapter = str(s.adapter_dir) if use_adapter and s.adapter_dir.exists() else None
        return HFChatLLM(base_model=s.base_model, adapter_dir=adapter)
    raise ValueError(f"backend desconhecido: {backend}")
