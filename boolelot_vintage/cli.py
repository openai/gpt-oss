"""Command-line interface for BooleLOT."""
import argparse

from .config import VintageConfig
from .engine import VintageEngine


def main():
    parser = argparse.ArgumentParser(description="BooleLOT vintage AI")
    parser.add_argument("--model", help="Model identifier or Talkie model alias")
    parser.add_argument("--backend", choices=("talkie", "transformers"))
    parser.add_argument("--period-start", type=int, default=1800)
    parser.add_argument("--period-end", type=int, default=1899)
    args = parser.parse_args()

    base = VintageConfig.from_env()
    config = VintageConfig(
        model_id=args.model or base.model_id,
        backend=args.backend or base.backend,
        period_start=args.period_start,
        period_end=args.period_end,
        max_new_tokens=base.max_new_tokens,
        temperature=base.temperature,
        top_p=base.top_p,
        repetition_penalty=base.repetition_penalty,
        device_map=base.device_map,
        torch_dtype=base.torch_dtype,
    )
    engine = VintageEngine(config)
    print("BooleLOT Vintage AI — enter /quit to exit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/quit", "/exit"}:
            break
        if not prompt:
            continue
        try:
            print("\nBooleLOT:", engine.generate(prompt).text)
        except Exception as exc:
            print(f"\nRuntime error: {exc}")


if __name__ == "__main__":
    main()
