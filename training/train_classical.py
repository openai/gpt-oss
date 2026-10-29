"""以已校習卷微調，另存附加權重。"""

import argparse
import json
import tempfile
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        messages = row.get("messages", [])
        if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{path} 中問答次序不合。")
        if any(not isinstance(item.get("content"), str) or not item["content"].strip() for item in messages):
            raise ValueError(f"{path} 中問答為空。")
        if not row.get("source_id") or not row.get("location"):
            raise ValueError(f"{path} 中缺少出處。")
        if type(row.get("knowledge_year")) is not int or not 1 <= row["knowledge_year"] <= 1899:
            raise ValueError(f"{path} 中年代不合。")
    return rows


def main():
    parser = argparse.ArgumentParser(description="訓練古典學者之 LoRA 附加權重。")
    parser.add_argument("--data", type=Path, default=Path("data/classical/training"))
    parser.add_argument("--output", type=Path, default=Path("data/classical/model-adapter"))
    parser.add_argument("--base-model", default="unsloth/gpt-oss-20b")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--allow-small-dataset", action="store_true", help="僅供小樣本流程試驗")
    parser.add_argument("--dry-run", action="store_true", help="只校語料，不載模型")
    args = parser.parse_args()
    # 習卷必須與現行已校札記逐條相符，防止夾入未審材料。
    from training.prepare_corpus import build_corpus
    with tempfile.TemporaryDirectory() as temp:
        verified = Path(temp)
        build_corpus(Path("docs/classical_sources.json"), verified)
        for name in ("train.jsonl", "eval.jsonl"):
            if (args.data / name).read_bytes() != (verified / name).read_bytes():
                parser.error(f"{name} 與已校札記不符；請重新編卷，不可加入未審材料。")
    train = read_rows(args.data / "train.jsonl")
    evaluation = read_rows(args.data / "eval.jsonl")
    if not train or not evaluation:
        parser.error("習卷及考卷不可為空。")
    if {r["source_id"] for r in train} & {r["source_id"] for r in evaluation}:
        parser.error("習卷與考卷不可含同一書源。")
    print(f"習卷 {len(train)} 則；考卷 {len(evaluation)} 則。")
    if args.dry_run:
        return
    if len(train) < 32 and not args.allow_small_dataset:
        parser.error("習題不足 32 則；須先擴充校訂語料。若僅驗證管線，明示 --allow-small-dataset。")
    if args.max_steps <= 0:
        parser.error("max-steps 須為正數。")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("輸出目錄已有內容，請選擇新的目錄。")
    try:
        from unsloth import FastLanguageModel
        import torch
        from unsloth.chat_templates import train_on_responses_only
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        parser.error(f"缺少訓練套件：{exc}；請依文件在 GPU 環境安裝。")
    if not torch.cuda.is_available():
        parser.error("未找到 CUDA GPU；不能在此環境訓練 gpt-oss-20b。")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model, max_seq_length=2048, load_in_4bit=True, offload_embedding=True)
    model = FastLanguageModel.get_peft_model(
        model, r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)
    def dataset(rows):
        texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False) for r in rows]
        if not all("<|start|>user<|message|>" in t and "<|start|>assistant<|channel|>final<|message|>" in t for t in texts):
            raise ValueError("模型的對話格式與預期不符；不可靜默套用錯誤的回覆遮罩。")
        if any(len(tokenizer(t, add_special_tokens=False)["input_ids"]) > 2048 for t in texts):
            raise ValueError("習題超過上下文長度；請先縮短題目，勿靜默截斷答案。")
        return Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, processing_class=tokenizer,
        train_dataset=dataset(train), eval_dataset=dataset(evaluation),
        args=SFTConfig(output_dir=str(args.output / "checkpoints"), dataset_text_field="text",
            max_length=2048, per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=args.max_steps, learning_rate=2e-4, logging_steps=5, save_strategy="no",
            report_to="none", seed=3407))
    trainer = train_on_responses_only(trainer, instruction_part="<|start|>user<|message|>",
                                      response_part="<|start|>assistant<|channel|>final<|message|>")
    trainer.train()
    metrics = trainer.evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "training_manifest.json").write_text(json.dumps({
        "base_model": args.base_model, "train_sources": sorted({r['source_id'] for r in train}),
        "eval_sources": sorted({r['source_id'] for r in evaluation}), "train_count": len(train),
        "eval_count": len(evaluation), "max_steps": args.max_steps, "metrics": metrics,
        "status": "adapter_only; not evaluated for historical exclusion or programming"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"附加權重已存 {args.output}；仍須進行獨立驗收。")


if __name__ == "__main__":
    main()
