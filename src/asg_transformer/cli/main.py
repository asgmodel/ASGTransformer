from __future__ import annotations

import argparse
import json

from asg_transformer import __version__
from asg_transformer.config import settings
from asg_transformer.core.service import get_service


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="asg",
        description="ASGTransformer command-line interface",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    scenario = commands.add_parser("scenario", help="Generate a defensive scenario")
    scenario.add_argument("text")
    scenario.add_argument("--language", choices=["en", "ar"], default="en")
    scenario.add_argument("--max-new-tokens", type=int, default=settings.max_new_tokens)
    scenario.add_argument("--sample", action="store_true")
    scenario.add_argument("--temperature", type=float, default=0.7)
    scenario.add_argument("--top-p", type=float, default=0.9)

    embedding = commands.add_parser("embed", help="Create a semantic embedding")
    embedding.add_argument("text")

    commands.add_parser("doctor", help="Print runtime configuration")
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "asg_transformer.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    elif args.command == "scenario":
        options = {
            "language": args.language,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.sample,
        }
        if args.sample:
            options.update(temperature=args.temperature, top_p=args.top_p)
        result = get_service().generate_scenario(args.text, **options)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "embed":
        vector = get_service().encode(args.text)
        print(json.dumps({"dimension": len(vector), "embedding": vector}, indent=2))
    elif args.command == "doctor":
        print(
            json.dumps(
                {
                    "version": __version__,
                    "model_id": settings.model_id,
                    "model_dir": str(settings.model_dir),
                    "model_source": settings.model_source,
                    "trust_remote_code": settings.trust_remote_code,
                    "torch_dtype": settings.torch_dtype,
                    "device_map": settings.device_map,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
