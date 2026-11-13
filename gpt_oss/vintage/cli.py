"""Interactive terminal client for vintage model profiles."""
import argparse
from .engine import VintageEngine
from .registry import MODEL_REGISTRY


def main():
    choices = sorted(key for key, value in MODEL_REGISTRY.items() if value.task == "generation")
    parser = argparse.ArgumentParser(description="BooleLOT vintage model playground")
    parser.add_argument("--model", choices=choices, default="talkie-1930-it")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    engine = VintageEngine(args.model, max_new_tokens=args.max_tokens, temperature=args.temperature)
    system = (
        "You are BooleLOT, a historically grounded assistant. Use period-appropriate diction "
        "without parody. Distinguish documented facts from conjecture. Do not invent quotations "
        "or sources. Never claim that prompting removes knowledge from model weights."
    )
    print(f"BooleLOT Vintage — {engine.profile.model_id} (type /quit to exit)")
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
            print("\nBooleLOT:", engine.generate(prompt, system=system).text)
        except Exception as exc:
            print(f"\nInference error: {exc}")


if __name__ == "__main__":
    main()
