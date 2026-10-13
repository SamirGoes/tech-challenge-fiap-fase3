import argparse
import json
from pathlib import Path

from medassist.assistente import MODELO_BASE, ADAPTADOR


def ler(caminho):
    return [json.loads(l) for l in Path(caminho).read_text(encoding="utf-8").splitlines() if l]


def tokenizar(exemplo, tokenizer, max_len=1024):
    mensagens = exemplo["messages"]
    prompt = tokenizer.apply_chat_template(mensagens[:-1], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt + mensagens[-1]["content"] + tokenizer.eos_token, add_special_tokens=False,
                    truncation=True, max_length=max_len)["input_ids"]
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * min(n_prompt, len(ids)) + ids[n_prompt:]}


def treinar(pasta_dados="dados", saida=ADAPTADOR, epocas=3):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quantizacao = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, quantization_config=quantizacao, device_map="auto")
    modelo = prepare_model_for_kbit_training(modelo)
    modelo = get_peft_model(modelo, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    modelo.print_trainable_parameters()

    def carregar(nome):
        return Dataset.from_list(ler(f"{pasta_dados}/{nome}.jsonl")).map(
            lambda ex: tokenizar(ex, tokenizer), remove_columns=["messages", "tipo"])

    trainer = Trainer(
        model=modelo,
        train_dataset=carregar("train"),
        eval_dataset=carregar("val"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(output_dir=f"{saida}/checkpoints", num_train_epochs=epocas, learning_rate=2e-4,
                               per_device_train_batch_size=4, gradient_accumulation_steps=4, fp16=True,
                               eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                               logging_steps=5, report_to="none"),
    )
    trainer.train()
    modelo.save_pretrained(saida)
    tokenizer.save_pretrained(saida)
    Path(saida, "historico.json").write_text(json.dumps(trainer.state.log_history, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=float, default=3)
    treinar(epocas=parser.parse_args().epocas)
