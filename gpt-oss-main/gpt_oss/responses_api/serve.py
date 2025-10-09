# torchrun --nproc-per-node=4 serve.py

import argparse

import uvicorn
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
)

from .api_server import create_api_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Responses API server")
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
        default="~/model",
        required=False,
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--inference-backend",
        metavar="BACKEND",
        type=str,
        help="Inference backend to use",
        # default to metal on macOS, triton on other platforms
        default="metal" if __import__("platform").system() == "Darwin" else "triton",
    )
    args = parser.parse_args()

    if args.inference_backend == "triton":
        from .inference.triton import setup_model
    elif args.inference_backend == "stub":
        from .inference.stub import setup_model
    elif args.inference_backend == "metal":
        from .inference.metal import setup_model
    elif args.inference_backend == "ollama":
        from .inference.ollama import setup_model
    elif args.inference_backend == "vllm":
        from .inference.vllm import setup_model
    elif args.inference_backend == "transformers":
        from .inference.transformers import setup_model
    else:
        raise ValueError(f"Invalid inference backend: {args.inference_backend}")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    infer_next_token = setup_model(args.checkpoint)
    uvicorn.run(create_api_server(infer_next_token, encoding), port=args.port)
