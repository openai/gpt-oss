"""以已校史札與符術課微調，另存附加權重。

習卷必須與現行兩份校訂目錄逐位元相符，防止夾入未審材料或後世史事。
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from gpt_oss.chronology import ALLOWED_LANGUAGES, ChronologyGuard, post_cutoff_years
from training.prepare_corpus import (CLASSICAL_CATALOGUE, CODING_CATALOGUE,
                                     ROOT, build_corpus, sha256)


# Minimum data-volume gates for a candidate adapter. These are safeguards,
# not a guarantee of accuracy or coding competence.
MIN_HISTORICAL_TRAIN = 1_000
MIN_PROGRAMMING_TRAIN = 1_000
MIN_HISTORICAL_EVAL = 100
MIN_PROGRAMMING_EVAL = 100
MIN_HISTORICAL_TRAIN_SOURCES = 25


def read_rows(path: Path) -> list[dict]:
    guard = ChronologyGuard()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        messages = row.get("messages", [])
        if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"{path} 中問答次序不合。")
        if any(not isinstance(item.get("content"), str) or not item["content"].strip() for item in messages):
            raise ValueError(f"{path} 中問答為空。")
        if not row.get("source_id") or not row.get("location"):
            raise ValueError(f"{path} 中缺少出處。")
        domain = row.get("knowledge_domain")
        if domain == "historical":
            year = row.get("knowledge_year")
            if type(year) is not int or not 1 <= year <= 1899 or row.get("cutoff_exception") is not False:
                raise ValueError(f"{path} 中史札年代不合。")
        elif domain == "programming":
            if row.get("cutoff_exception") is not True or row.get("knowledge_year") is not None:
                raise ValueError(f"{path} 中程式特例標記不合。")
            if set(row.get("languages", [])) - ALLOWED_LANGUAGES:
                raise ValueError(f"{path} 中有未許之程式語。")
            code_file = ROOT / row.get("code_file", "")
            if not code_file.is_file() or sha256(code_file) != row.get("code_sha256"):
                raise ValueError(f"{path} 中程式例與校訂本不符。")
            prose = "\n".join(item["content"] for item in messages)
            if guard.reasons(prose, programming=True) or post_cutoff_years(prose):
                raise ValueError(f"{path} 中疑有年界後世事。")
        else:
            raise ValueError(f"{path} 中知識域未明。")
    return rows


def main():
    parser = argparse.ArgumentParser(description="訓練古典學者之 LoRA 附加權重。")
    parser.add_argument("--data", type=Path, default=Path("data/classical/training"))
    parser.add_argument("--output", type=Path, default=Path("data/classical/model-adapter"))
    parser.add_argument("--classical-catalogue", type=Path, default=CLASSICAL_CATALOGUE)
    parser.add_argument("--coding-catalogue", type=Path, default=CODING_CATALOGUE)
    parser.add_argument("--base-model", default="unsloth/gpt-oss-20b")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--allow-small-dataset", action="store_true", help="僅供小樣本流程試驗")
    parser.add_argument("--dry-run", action="store_true", help="只校語料，不載模型")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        verified = Path(temp)
        build_corpus(args.classical_catalogue, args.coding_catalogue, verified)
        for name in ("train.jsonl", "eval.jsonl", "manifest.json"):
            if (args.data / name).read_bytes() != (verified / name).read_bytes():
                parser.error(f"{name} 與已校目錄不符；請重新編卷，不可加入未審材料。")
    train = read_rows(args.data / "train.jsonl")
    evaluation = read_rows(args.data / "eval.jsonl")
    if not train or not evaluation:
        parser.error("習卷及考卷不可為空。")
    if {r["source_id"] for r in train} & {r["source_id"] for r in evaluation}:
        parser.error("習卷與考卷不可含同一書源或題源。")
    for name, rows in (("習卷", train), ("考卷", evaluation)):
        for domain in ("historical", "programming"):
            if not any(r["knowledge_domain"] == domain for r in rows):
                parser.error(f"{name} 須含 {domain} 域之題目。")

    counts = {
        "historical_train": sum(r["knowledge_domain"] == "historical" for r in train),
        "programming_train": sum(r["knowledge_domain"] == "programming" for r in train),
        "historical_eval": sum(r["knowledge_domain"] == "historical" for r in evaluation),
        "programming_eval": sum(r["knowledge_domain"] == "programming" for r in evaluation),
    }
    historical_sources = {r["source_id"] for r in train if r["knowledge_domain"] == "historical"}
    minimums = {
        "historical_train": MIN_HISTORICAL_TRAIN,
        "programming_train": MIN_PROGRAMMING_TRAIN,
        "historical_eval": MIN_HISTORICAL_EVAL,
        "programming_eval": MIN_PROGRAMMING_EVAL,
    }
    shortages = [f"{name}: {counts[name]} / {minimum}"
                 for name, minimum in minimums.items() if counts[name] < minimum]
    if len(historical_sources) < MIN_HISTORICAL_TRAIN_SOURCES:
        shortages.append(
            f"historical_train_sources: {len(historical_sources)} / {MIN_HISTORICAL_TRAIN_SOURCES}"
        )

    print(
        f"習卷 {len(train)} 則（史事 {counts['historical_train']}、程式 {counts['programming_train']}）；"
        f"考卷 {len(evaluation)} 則（史事 {counts['historical_eval']}、"
        f"程式 {counts['programming_eval']}）。"
    )
    if args.dry_run:
        if shortages:
            print("訓練資格：未通過資料量門檻；目前只可作流程試驗。")
            print("\n".join(shortages))
        else:
            print("訓練資格：資料量門檻通過；仍須完成人工審核與真實模型評測。")
        return
    if len(train) < 32 and not args.allow_small_dataset:
        parser.error("習題不足 32 則；須先擴充校訂語料。若僅驗證管線，明示 --allow-small-dataset。")
    if shortages and not args.allow_small_dataset:
        parser.error(
            "資料不足以訓練候選模型；不得把小樣本適配器稱為成品。\n"
            + "\n".join(shortages)
            + "\\n只有流程試驗可明示 --allow-small-dataset。"
        )
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
        model, r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    def dataset(rows):
        texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                               add_generation_prompt=False) for r in rows]
        if not all("<|start|>user<|message|>" in t
                   and "<|start|>assistant<|channel|>final<|message|>" in t for t in texts):
            raise ValueError("模型的對話格式與預期不符；不可靜默套用錯誤的回覆遮罩。")
        if any(len(tokenizer(t, add_special_tokens=False)["input_ids"]) > 2048 for t in texts):
            raise ValueError("習題超過上下文長度；請先縮短題目，勿靜默截斷答案。")
        return Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(model=model, processing_class=tokenizer,
        train_dataset=dataset(train), eval_dataset=dataset(evaluation),
        args=SFTConfig(output_dir=str(args.output / "checkpoints"), dataset_text_field="text",
            max_length=2048, per_device_train_batch_size=1, gradient_accumulation_steps=4,
            max_steps=args.max_steps, learning_rate=1e-4, warmup_ratio=0.03,
            weight_decay=0.01, logging_steps=20, eval_strategy="steps",
            eval_steps=100, save_strategy="no", report_to="none", seed=3407))
    trainer = train_on_responses_only(trainer, instruction_part="<|start|>user<|message|>",
                                      response_part="<|start|>assistant<|channel|>final<|message|>")
    trainer.train()
    metrics = trainer.evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (args.output / "training_manifest.json").write_text(json.dumps({
        "base_model": args.base_model,
        "historical_cutoff_year": 1899,
        "train_sources": sorted({r["source_id"] for r in train}),
        "eval_sources": sorted({r["source_id"] for r in evaluation}),
        "train_count": len(train), "eval_count": len(evaluation),
        "programming_train_count": sum(r["knowledge_domain"] == "programming" for r in train),
        "programming_eval_count": sum(r["knowledge_domain"] == "programming" for r in evaluation),
        "max_steps": args.max_steps, "metrics": metrics,
        "data_quality_gate": {
            "minimums": minimums,
            "observed": counts,
            "historical_train_sources": len(historical_sources),
            "passed": not shortages,
            "shortages": shortages,
        },
        "quality_tier": "experimental_small_data" if args.allow_small_dataset and shortages
                        else "candidate_requires_independent_evaluation",
        "knowledge_isolation": (
            "training data and response policy are domain-scoped; the base model may "
            "retain post-cutoff knowledge and this adapter does not prove unlearning"
        ),
        "status": "adapter_only; not a standalone model and not a verified release",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"附加權重已存 {args.output}；仍須以未見題目與年代防火牆獨立驗收。")


if __name__ == "__main__":
    main()
