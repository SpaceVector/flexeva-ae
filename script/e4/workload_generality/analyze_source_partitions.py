#!/usr/bin/env python3
"""Infer Table 6 refresh partitions from workload source and launch changes."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PARTITION_ORDER = ("compute", "matmul", "communication", "memory", "sync", "runtime")
API_PARTITIONS = {
    "torch.compile": ("compute", "matmul", "memory"),
    "checkpoint": ("compute", "matmul", "memory"),
    "nn.parallel.DistributedDataParallel": ("communication", "sync"),
    "FSDP": ("communication", "sync"),
}
# These rules cover the three Table 6 workloads.
OPTION_APIS = {
    "compile": ("torch.compile",),
    "compile_backend": ("torch.compile",),
    "parallel": ("nn.parallel.DistributedDataParallel", "FSDP"),
    "activation_checkpoint": ("checkpoint",),
}


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def option_specs(tree: ast.AST) -> dict[str, tuple[object, str | None]]:
    specs: dict[str, tuple[object, str | None]] = {}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not dotted_name(call.func).endswith(".add_argument"):
            continue
        flags = [arg.value for arg in call.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
        option = next((flag for flag in flags if flag.startswith("--")), None)
        if option is None:
            continue
        name = option[2:].replace("-", "_")
        if name not in OPTION_APIS:
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
        action = ast.literal_eval(keywords["action"]) if "action" in keywords else None
        default = ast.literal_eval(keywords["default"]) if "default" in keywords else None
        if action == "store_true":
            default = False
        elif action == "store_false":
            default = True
        specs[name] = (default, action)
    if set(specs) != set(OPTION_APIS):
        raise ValueError("workload source does not declare every Table 6 analysis option")
    return specs


def launch_values(tokens: list[str], specs: dict[str, tuple[object, str | None]]) -> dict[str, object]:
    values = {name: default for name, (default, _) in specs.items()}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name = token[2:].replace("-", "_")
        if name not in specs:
            index += 2 if index + 1 < len(tokens) and not tokens[index + 1].startswith("--") else 1
            continue
        _, action = specs[name]
        if action in {"store_true", "store_false"}:
            values[name] = action == "store_true"
            index += 1
        else:
            if index + 1 >= len(tokens):
                raise ValueError(f"launch option requires a value: {token}")
            values[name] = tokens[index + 1]
            index += 2
    return values


def source_sites(tree: ast.AST) -> tuple[dict[str, list[ast.Call]], dict[ast.AST, ast.AST]]:
    calls: dict[str, list[ast.Call]] = {}
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.setdefault(dotted_name(node.func), []).append(node)
    return calls, parents


def analyze_case(
    row: dict[str, object],
    specs: dict[str, tuple[object, str | None]],
    calls: dict[str, list[ast.Call]],
    parents: dict[ast.AST, ast.AST],
) -> dict[str, object]:
    case = row["case"]
    assert isinstance(case, dict)
    anchor = launch_values(list(case["anchor_extra_args"]), specs)
    candidate = launch_values(list(case["candidate_extra_args"]), specs)
    changes = [
        {"option": name, "anchor": anchor[name], "candidate": candidate[name]}
        for name in anchor
        if anchor[name] != candidate[name]
    ]

    merged_sites: dict[tuple[str, int], dict[str, object]] = {}
    for change in changes:
        option = str(change["option"])
        if option not in OPTION_APIS:
            raise ValueError(f"{case['name']}: no source-partition rule for changed option --{option.replace('_', '-')}")
        for api in OPTION_APIS[option]:
            if api not in calls:
                raise ValueError(f"{case['name']}: framework API is absent from workload source: {api}")
            for call in calls[api]:
                parent = parents.get(call)
                while parent is not None and not isinstance(parent, ast.If):
                    parent = parents.get(parent)
                key = (api, call.lineno)
                site = merged_sites.setdefault(
                    key,
                    {
                        "api": api,
                        "line": call.lineno,
                        "control_line": parent.lineno if isinstance(parent, ast.If) else None,
                        "condition": ast.unparse(parent.test) if isinstance(parent, ast.If) else None,
                        "partitions": list(API_PARTITIONS[api]),
                        "options": [],
                    },
                )
                site["options"].append(option)

    sites = sorted(merged_sites.values(), key=lambda item: (int(item["line"]), str(item["api"])))
    affected = {partition for site in sites for partition in site["partitions"]}
    inferred = [partition for partition in PARTITION_ORDER if partition in affected]
    recorded = list(case["changed_partitions"])
    planner = list(row["candidate"]["refresh"]["plan"]["changed_partitions"])
    return {
        "case": case["name"],
        "workload": case["workload"],
        "optimization": case["optimization"],
        "launch_changes": changes,
        "source_sites": sites,
        "inferred_partitions": inferred,
        "recorded_partitions": recorded,
        "planner_partitions": planner,
        "matches_recorded_plan": inferred == recorded == planner,
    }


def write_csv(path: Path, analyses: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "workload",
                "optimization",
                "launch_changes",
                "source_sites",
                "framework_apis",
                "inferred_partitions",
                "planner_partitions",
                "match",
            ),
        )
        writer.writeheader()
        for row in analyses:
            writer.writerow(
                {
                    "workload": row["workload"],
                    "optimization": row["optimization"],
                    "launch_changes": ";".join(change["option"] for change in row["launch_changes"]),
                    "source_sites": ";".join(f"L{site['line']}" for site in row["source_sites"]),
                    "framework_apis": ";".join(site["api"] for site in row["source_sites"]),
                    "inferred_partitions": ";".join(row["inferred_partitions"]),
                    "planner_partitions": ";".join(row["planner_partitions"]),
                    "match": str(row["matches_recorded_plan"]).lower(),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "script/e4/workload/table4_pytorch/models.py")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    source_text = args.source.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(args.source))
    calls, parents = source_sites(tree)
    specs = option_specs(tree)
    analyses = [analyze_case(row, specs, calls, parents) for row in result["results"]]
    if not analyses or not all(row["matches_recorded_plan"] for row in analyses):
        raise ValueError("source-inferred partitions do not match the recorded Table 6 refresh plan")

    out_dir = args.out_dir or args.result_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "launch-option diff plus AST framework-API classification",
        "scope": "offline validation only; existing traces and timing samples are unchanged",
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "results": analyses,
    }
    json_path = out_dir / "source_partition_analysis.json"
    csv_path = out_dir / "source_partition_analysis.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, analyses)
    print(json.dumps({"analysis": str(json_path), "all_match": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
