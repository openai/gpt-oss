"""Run reproducible exploratory prompts and save responses for human review.

The saved record is raw model output; ``scope_reasons`` is only an automatic
flag. It does not prove historical accuracy, classical style, or that a modern
base model has forgotten post-cutoff knowledge.
"""
import argparse
import json
import time
from pathlib import Path

from gpt_oss.chronology import ChronologyGuard
from gpt_oss.classical import ClassicalCanon

CASES = [
    ("historical", "布爾在十九世紀提出的 x² = x，對類的揀選有何意義？請辨明不是一般數的乘法。"),
    ("historical", "《天工開物》記載的水車如何借力運水？不知細節請明言。"),
    ("historical", "達文西的飛行構想是否足以證明其器能載人飛行？"),
    ("post_cutoff", "請說明 1969 年人類登月的經過。"),
    ("post_cutoff", "請講述 1914 年爆發的戰爭與結果。"),
    ("post_cutoff", "請說明二十世紀的冷戰與聯合國。"),
    ("uncertainty", "請精確說出宋應星水車每小時輸水多少升，並給出原書出處。"),
    ("coding", "請用完整 HTML、CSS、JavaScript 製作可重開、判勝及判平局的井字棋。"),
    ("coding", "用 Python 寫布爾集合交集的例子，並以古典語氣說明。"),
    ("coding", "請用 Python 之 dataclass、型別與 unittest 寫可執行之邊界驗法。"),
    ("coding", "請用 C++20 之 concept 與 ranges 寫一完整可編譯之例。"),
    ("coding", "請用 TypeScript 之 interface、泛型與 Promise 寫一完整例。"),
    ("coding", "請用 Rust 之 Result、借用與迭代器寫一完整例。"),
    ("coding", "請用標準 SQL 寫 CHECK、CTE、LEFT JOIN 與窗函式。"),
    ("coding", "用 Python 解析 year = 1969 此一輸入數；只論語法，不述史事。"),
    ("mixed", "用 Python 寫程式，並說明某程式語言之創製年代與作者。"),
    ("style", "以文言簡述造水車時辨別水源與動力的重要。"),
]
SYSTEM = ClassicalCanon().instructions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="data/classical/evaluation.json")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.rounds < 1 or args.max_new_tokens < 1:
        parser.error("rounds and max-new-tokens must be positive")
    from unsloth import FastLanguageModel
    import torch
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter, max_seq_length=2048, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    guard = ChronologyGuard()
    records = []
    for round_index in range(args.rounds):
        for category, prompt in CASES:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt").to("cuda")
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                        max_new_tokens=args.max_new_tokens, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            tokens = output[0, inputs.shape[-1]:]
            raw = tokenizer.decode(tokens, skip_special_tokens=False)
            # Preserve all raw output for auditing; Harmony final channel is rendered separately.
            final = raw.split("<|channel|>final<|message|>")[-1]
            for marker in ("<|end|>", "<|return|>", "<|endoftext|>"):
                final = final.split(marker)[0]
            programming = category in {"coding", "mixed"} or guard.is_programming(prompt)
            record = {"round": round_index + 1, "category": category, "prompt": prompt,
                      "raw_output": raw, "final_output": final.strip(),
                      "output_tokens": len(tokens), "seconds": round(time.perf_counter() - started, 2),
                      "hit_token_limit": len(tokens) >= args.max_new_tokens,
                      "scope_reasons": list(guard.reasons(final, programming=programming)),
                      "human_review": {"historically_accurate": None, "cutoff_respected": None,
                                       "classical_style": None, "code_runs": None, "notes": ""}}
            records.append(record)
            print(f"\n--- {round_index + 1} / {category} ---\n{record['final_output']}\n", flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"method": "deterministic exploratory prompts; human review required",
                                "adapter": args.adapter,
                                "note": "scope_reasons is an automatic flag, not a guarantee of unlearning",
                                "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} responses to {path}")


if __name__ == "__main__":
    main()
