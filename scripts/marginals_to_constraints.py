"""
Convert MWEM marginals ({"marginals": [...]}) to constraint_eval.py schema
({"constraints": [...]}) so we can reuse the L1 evaluator for MWEM-only samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def convert(mwem: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    constraints = []
    for m in mwem.get("marginals", []):
        constraints.append(
            {
                "columns": m.get("columns", []),
                "target_probs": m.get("probs", {}),
            }
        )
    return {"constraints": constraints}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mwem_marginals.json into constraint_eval.py format."
    )
    parser.add_argument("--marginals", required=True, help="Path to mwem_marginals.json")
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the converted constraints JSON.",
    )
    args = parser.parse_args()

    mwem = json.loads(Path(args.marginals).read_text())
    converted = convert(mwem)
    Path(args.output).write_text(json.dumps(converted, indent=2))
    print(f"Wrote constraints JSON to {args.output}")


if __name__ == "__main__":
    main()
