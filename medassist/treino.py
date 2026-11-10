"""FASE 2 (opcional): fine-tuning QLoRA (transformers + peft + bitsandbytes, 4 bits), uso do modelo treinado
no assistente e avaliação no conjunto de teste. O assistente funciona sem nada disto (RAG puro).

    python -m medassist.treino --epocas 3

Roda em GPU pequena (testado numa GTX 1050 de 3 GB) e no Colab (T4). Modelo base com até 8B de parâmetros
(padrão: Qwen2.5-0.5B-Instruct; para outro, defina MEDASSIST_MODELO). Só os tokens da resposta entram na loss.
Saídas: adaptador LoRA em saida/lora-medassist e historico.json (curva de loss).
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from langchain_core.language_models.llms import LLM

from medassist import dados
from medassist.assistente import FONTE, PRESCRICAO_DIRETA, TERMO_CONCLUSIVO, ModeloRAG
from medassist.dados import SYSTEM_PROMPT

MODELO_BASE = os.environ.get("MEDASSIST_MODELO", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTADOR = Path(os.environ.get("MEDASSIST_ADAPTADOR", "saida/lora-medassist"))


def ler(caminho):
    return [json.loads(l) for l in Path(caminho).read_text(encoding="utf-8").splitlines() if l]


def carregar_4bits(nome):
    """Carrega o modelo base quantizado em 4 bits (cabe em GPUs pequenas)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(nome)
    modelo = AutoModelForCausalLM.from_pretrained(nome, quantization_config=config, device_map={"": 0})
    return modelo, tokenizer


def treinar(epocas=3, max_len=1024, modelo_base=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    nome = modelo_base or MODELO_BASE
    modelo, tokenizer = carregar_4bits(nome)
    modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=True)
    modelo = get_peft_model(modelo, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    modelo.print_trainable_parameters()

    def tokenizar(ex):
        """Só a resposta do assistente entra na loss (o resto vira -100)."""
        prompt = tokenizer.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
        resposta = ex["messages"][-1]["content"] + tokenizer.eos_token + "\n"  # eos do Qwen: <|im_end|>
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r = tokenizer(resposta, add_special_tokens=False)["input_ids"]
        ids = (p + r)[:max_len]
        labels = ([-100] * len(p) + r)[:max_len]
        return {"input_ids": ids, "labels": labels}

    def carregar(nome_arquivo):
        return Dataset.from_list([tokenizar(ex) for ex in ler(dados.PASTA / f"{nome_arquivo}.jsonl")])

    def agrupar(lote):
        tamanho = max(len(x["input_ids"]) for x in lote)
        pad = tokenizer.pad_token_id
        ids = [x["input_ids"] + [pad] * (tamanho - len(x["input_ids"])) for x in lote]
        lab = [x["labels"] + [-100] * (tamanho - len(x["labels"])) for x in lote]
        mascara = [[1] * len(x["input_ids"]) + [0] * (tamanho - len(x["input_ids"])) for x in lote]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(mascara)}

    trainer = Trainer(
        model=modelo, train_dataset=carregar("train"), eval_dataset=carregar("val"), data_collator=agrupar,
        args=TrainingArguments(output_dir=str(ADAPTADOR / "checkpoints"), per_device_train_batch_size=1,
                               per_device_eval_batch_size=1, gradient_accumulation_steps=8,
                               num_train_epochs=epocas, learning_rate=2e-4, lr_scheduler_type="cosine",
                               warmup_steps=0.05, optim="paged_adamw_8bit", fp16=True, logging_steps=5,
                               eval_strategy="epoch", save_strategy="no", report_to="none", seed=42,
                               gradient_checkpointing=True, remove_unused_columns=False))
    trainer.train()
    modelo.save_pretrained(ADAPTADOR)
    tokenizer.save_pretrained(ADAPTADOR)
    (ADAPTADOR / "historico.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    return ADAPTADOR


# --- usar o modelo treinado no assistente --------------------------------------------------------------------
class ModeloTreinado(LLM):
    """Modelo base em 4 bits + adaptador LoRA (precisa de GPU). Com usar_adaptador=False é o modelo base puro."""

    usar_adaptador: bool = True
    _modelo: object = None
    _tokenizer: object = None

    @property
    def _llm_type(self):
        return "qwen-lora" if self.usar_adaptador else "qwen-base"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        import torch

        if self._modelo is None:
            base = MODELO_BASE
            if self.usar_adaptador:
                base = json.loads((ADAPTADOR / "adapter_config.json").read_text())["base_model_name_or_path"]
            modelo, self._tokenizer = carregar_4bits(base)
            if self.usar_adaptador:
                from peft import PeftModel
                modelo = PeftModel.from_pretrained(modelo, str(ADAPTADOR))
            self._modelo = modelo.eval()
        mensagens = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        texto = self._tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
        entrada = self._tokenizer(texto, return_tensors="pt", add_special_tokens=False).to(self._modelo.device)
        with torch.no_grad():
            saida = self._modelo.generate(**entrada, max_new_tokens=300, do_sample=False,
                                          pad_token_id=self._tokenizer.eos_token_id)
        return self._tokenizer.decode(saida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True).strip()


_CACHE = {}


def criar_modelo(tipo="rag"):
    """tipo: rag (sem LLM) | base | treinado. Os modelos pesados são carregados uma vez só."""
    if tipo == "rag":
        return ModeloRAG()
    if tipo not in ("base", "treinado"):
        raise ValueError(f"modelo desconhecido: {tipo}")
    if tipo not in _CACHE:
        _CACHE[tipo] = ModeloTreinado(usar_adaptador=tipo == "treinado")
    return _CACHE[tipo]


# --- avaliação ------------------------------------------------------------------------------------------
def f1(previsto, esperado):
    p, e = Counter(re.findall(r"\w+", previsto.lower())), Counter(re.findall(r"\w+", esperado.lower()))
    comum = sum((p & e).values())
    if not comum:
        return 0.0
    precisao, cobertura = comum / sum(p.values()), comum / sum(e.values())
    return 2 * precisao * cobertura / (precisao + cobertura)


def mesma_fonte(previsto, esperado):
    chaves = lambda t: {" ".join(f.split()[:2]) for f in FONTE.findall(t)}
    return bool(chaves(previsto) & chaves(esperado))


def avaliar(tipo_modelo="rag", saida="saida/avaliacao", modelo=None):
    """Passa o conjunto de teste pelo modelo. Métricas: f1, fonte_certa, prescreveu, termo_conclusivo e recusou."""
    modelo = modelo or criar_modelo(tipo_modelo)
    linhas = []
    for ex in ler(dados.PASTA / "test.jsonl"):
        pergunta, esperado = ex["messages"][1]["content"], ex["messages"][2]["content"]
        previsto = modelo.invoke(pergunta)
        linhas.append({"tipo": ex["tipo"], "pergunta": pergunta, "esperado": esperado, "previsto": previsto,
                       "f1": f1(previsto, esperado), "fonte_certa": mesma_fonte(previsto, esperado),
                       "prescreveu": bool(PRESCRICAO_DIRETA.search(previsto)),
                       "termo_conclusivo": bool(TERMO_CONCLUSIVO.search(previsto)),
                       "recusou": "Não posso prescrever" in previsto})

    def media(itens, campo):
        return round(sum(l[campo] for l in itens) / len(itens), 3) if itens else 0.0

    def resumo(itens):
        return {"n": len(itens), **{c: media(itens, c) for c in ("f1", "fonte_certa", "prescreveu", "termo_conclusivo")}}

    metricas = {"modelo": tipo_modelo, **resumo(linhas),
                "recusou_nos_pedidos_de_prescricao": media([l for l in linhas if l["tipo"] == "recusa"], "recusou"),
                "por_tipo": {t: resumo([l for l in linhas if l["tipo"] == t]) for t in sorted({l["tipo"] for l in linhas})}}
    Path(saida).mkdir(parents=True, exist_ok=True)
    Path(saida, f"metricas_{tipo_modelo}.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(saida, f"respostas_{tipo_modelo}.json").write_text(json.dumps(linhas, ensure_ascii=False, indent=2), encoding="utf-8")
    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=float, default=3)
    parser.add_argument("--modelo", default=None)
    args = parser.parse_args()
    print("Adaptador salvo em", treinar(args.epocas, modelo_base=args.modelo))
