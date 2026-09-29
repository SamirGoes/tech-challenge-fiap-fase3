"""Fine-tuning com LoRA/QLoRA sobre um modelo instruct aberto.

Pensado para rodar no Google Colab (GPU T4, 16 GB). Exemplo:

    python -m medassist.finetune.train_lora \
        --base-model Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --out outputs/lora-medassist

Somente os tokens da resposta do assistente entram na loss (as partes de
sistema e usuário são mascaradas com -100).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medassist.config import get_settings


def _ler(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tokenizar(exemplo: dict, tokenizer, max_len: int) -> dict:
    msgs = exemplo["messages"]
    prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    completo = prompt + msgs[-1]["content"] + tokenizer.eos_token
    ids = tokenizer(completo, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    labels = [-100] * min(n_prompt, len(ids)) + ids[n_prompt:]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default=s.base_model)
    ap.add_argument("--data-dir", type=Path, default=s.processed_dir)
    ap.add_argument("--out", type=Path, default=s.adapter_dir)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-4bit", action="store_true", help="desativa QLoRA (usa fp16/bf16)")
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.pad_token = tok.pad_token or tok.eos_token

    quant = None
    if not args.no_4bit and torch.cuda.is_available():
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant,
                                                 torch_dtype=torch.float16, device_map="auto")
    if quant:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    def carregar(nome: str) -> Dataset:
        return Dataset.from_list(_ler(args.data_dir / f"{nome}.jsonl")).map(
            lambda ex: tokenizar(ex, tok, args.max_len), remove_columns=["messages", "tipo", "fonte"])

    train, val = carregar("train"), carregar("val")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            eval_strategy="epoch", save_strategy="epoch", logging_steps=5, fp16=torch.cuda.is_available(),
            lr_scheduler_type="cosine", warmup_ratio=0.05, report_to="none", load_best_model_at_end=True,
        ),
        train_dataset=train, eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "train_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"Adaptador LoRA salvo em {args.out}")


if __name__ == "__main__":
    main()
