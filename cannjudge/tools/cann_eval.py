#!/usr/bin/env python3
"""Run a recipe from a local evaluator binding without an ACW service."""

import argparse
import json
import os
import re
from pathlib import Path
import sys
import tempfile
import time

TOOLS = Path(__file__).resolve().parent


def document(path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    try:
        import yaml
    except ImportError:
        raise SystemExit("YAML bindings require PyYAML: python3 -m pip install PyYAML") from None
    return yaml.safe_load(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("build", "health", "check", "measure", "profile"))
    ap.add_argument("--binding", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=Path.cwd())
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--case-id", action="append", default=[])
    ap.add_argument("--timeout", type=int, help="execution seconds after acquiring the lock")
    ap.add_argument("--cann-home", type=Path, help="override the binding's CANN installation")
    ap.add_argument("--physical-device", type=int)
    ap.add_argument("--logical-device", type=int)
    args = ap.parse_args()

    base = args.binding.resolve(strict=True).parent
    binding = document(args.binding)
    paths = {name: (base / binding[name + "_ref"]).resolve(strict=True)
             for name in ("cases", "golden", "environment", "score_adapter")}
    adapter = document(paths["score_adapter"])
    recipe = adapter.get("commands", {}).get(args.phase)
    if adapter.get("evaluation_scope") != "local" or not recipe:
        ap.error("binding must name a local recipe for the requested phase")
    case_ids = [case["id"] for case in document(paths["cases"])["cases"]]
    selected = list(dict.fromkeys(args.case_id or case_ids))
    if not selected or set(selected) - set(case_ids):
        ap.error("select existing local case IDs")

    source = args.source.resolve(strict=True)
    run = args.run_dir.resolve()
    build = run / "build"
    if args.phase not in {"build", "health"} and not build.is_dir():
        ap.error("build this candidate in the given run directory first")
    build.mkdir(parents=True, exist_ok=True)
    temporary = run / "tmp"
    temporary.mkdir(exist_ok=True)
    cache = run / "cache"
    cache.mkdir(exist_ok=True)
    config = document(paths["environment"])
    toolkit = (args.cann_home or Path(config["cann"]["home"])).resolve()
    cann_env = toolkit / "set_env.sh" if args.cann_home else Path(config["cann"]["set_env"])
    config["cann"].update(home=str(toolkit), set_env=str(cann_env))
    if "harness_ref" in config:
        config["harness_ref"] = str((paths["environment"].parent / config["harness_ref"]).resolve())
    for key, value in (("physical_device", args.physical_device), ("logical_device", args.logical_device)):
        if value is not None:
            if value < 0:
                ap.error("device IDs must be nonnegative")
            config["npu"][key] = value
    if args.physical_device is not None:
        config["npu"]["device_id"] = args.physical_device
    # JSON is valid YAML and preserves the adapter's original environment keys.
    with tempfile.NamedTemporaryFile(mode="w", dir=run, prefix="environment-", suffix=".json", delete=False) as stream:
        json.dump(config, stream)
        runtime_environment = stream.name
    result = run / f"raw-{args.phase}-{time.time_ns()}.json"
    replacements = dict(binding_dir=str(base), python=sys.executable, cann_home=str(toolkit), cann_env=str(cann_env),
                        source=str(source), build=str(build), result=str(result), cases=str(paths["cases"]),
                        golden=str(paths["golden"]), environment=runtime_environment, job_id=result.stem)
    pattern = re.compile(r"\{(" + "|".join(replacements) + r")\}")
    argv = [pattern.sub(lambda match: replacements[match[1]], part) for part in recipe]
    timeout = args.timeout
    if timeout is None and args.phase != "build":
        timeout = adapter.get("timeout_seconds", 600)
    if timeout is not None:
        if not isinstance(timeout, int) or timeout < 1:
            ap.error("timeout must be a positive number of seconds")
        argv[:0] = ["timeout", "--verbose", "--kill-after=5s", str(timeout)]
    if args.phase != "build":
        argv[:0] = [str(TOOLS / "npu_lock.sh")]
    env = dict(os.environ, ACW_CASE_IDS=json.dumps(selected), ACW_SOURCE=str(source),
               ACW_RESULT=str(result), ACW_BUILD=str(build), TMPDIR=str(temporary),
               XDG_CACHE_HOME=str(cache), ASCEND_CACHE_PATH=str(cache / "cann"),
               ASCEND_WORK_PATH=str(run / "cann_work"))
    print(f"{args.phase}: {build}  result: {result}", flush=True)
    os.chdir(build)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
