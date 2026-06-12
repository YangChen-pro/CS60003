from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .secrets import has_any_key, load_env_file
from .utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--secret-env", default=".helloagents/secrets/hw3.env")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    load_env_file(args.secret_env)
    if not has_any_key(["MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN", "MODELSCOPE_SDK_TOKEN"]):
        raise RuntimeError("ModelScope token not found in environment")
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(
        access_token=(
            __import__("os").environ.get("MODELSCOPE_API_TOKEN")
            or __import__("os").environ.get("MODELSCOPE_TOKEN")
            or __import__("os").environ.get("MODELSCOPE_SDK_TOKEN")
        )
    )
    model_dir = Path(args.model_dir)
    files = [path for path in model_dir.rglob("*") if path.is_file()]
    for path in tqdm(files, desc=f"upload {args.repo_id}"):
        rel = path.relative_to(model_dir).as_posix()
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=args.repo_id,
            repo_type="model",
        )
    result = {"repo_id": args.repo_id, "model_dir": str(model_dir.resolve()), "files": len(files)}
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
