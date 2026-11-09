"""Fine-tuning com Unsloth (QLoRA em 4 bits). Pensado para o Google Colab (GPU T4).

    python -m medassist.treino --epocas 3

Modelo base com até 8B de parâmetros (padrão: Qwen2.5-3B-Instruct em 4 bits). Só os tokens da resposta
entram na loss (`train_on_responses_only`). Saídas: adaptador LoRA em saida/lora-medassist e historico.json (loss).
"""

import argparse
import json
from pathlib import Path

from medassist.config import config


def ler(caminho) -> list[dict]:
    return [json.loads(l) for l in Path(caminho).read_text(encoding="utf-8").splitlines() if l]


def treinar(pasta_dados=None, saida=None, modelo_base=None, epocas: float = 3, max_len: int = 2048) -> Path:
    from unsloth import FastLanguageModel, is_bfloat16_supported  # importar antes de transformers/trl
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    c = config()
    pasta_dados, saida = Path(pasta_dados or c.pasta_dados), Path(saida or c.adaptador)
    modelo_base = modelo_base or c.modelo_base

    modelo, tokenizer = FastLanguageModel.from_pretrained(model_name=modelo_base, max_seq_length=max_len,
                                                          load_in_4bit=True)
    modelo = FastLanguageModel.get_peft_model(
        modelo, r=16, lora_alpha=32, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        random_state=42)
    modelo.print_trainable_parameters()

    def carregar(nome: str) -> Dataset:
        textos = [tokenizer.apply_chat_template(ex["messages"], tokenize=False) for ex in ler(pasta_dados / f"{nome}.jsonl")]
        return Dataset.from_dict({"text": textos})

    trainer = SFTTrainer(
        model=modelo, tokenizer=tokenizer, train_dataset=carregar("train"), eval_dataset=carregar("val"),
        args=SFTConfig(
            output_dir=str(saida / "checkpoints"), dataset_text_field="text", max_seq_length=max_len,
            per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=epocas,
            learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, optim="adamw_8bit",
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=5,
            eval_strategy="epoch", save_strategy="no", report_to="none", seed=42))
    # Qwen usa o formato ChatML; para outro modelo, ajuste os dois marcadores abaixo.
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    trainer.train()
    modelo.save_pretrained(saida)
    tokenizer.save_pretrained(saida)
    (saida / "historico.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    return saida


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=float, default=3)
    parser.add_argument("--modelo", default=None)
    args = parser.parse_args()
    print("Adaptador salvo em", treinar(modelo_base=args.modelo, epocas=args.epocas))
