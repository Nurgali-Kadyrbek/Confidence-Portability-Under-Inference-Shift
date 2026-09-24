"""Single-observation adapter for the pinned official StyleMBPP evaluator.

This module is executed only inside the locked evaluation container. Its input
and output are one JSON object on standard input/output.
"""

from __future__ import annotations

import json
import sys
from typing import Any

sys.path.insert(0, "/opt/evaluator/code_bench")

from coding_vis_utils import convert_for_evaluation
from eval import evaluate_code


def score(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload["response"]
    benchmark = payload["scoring_payload"]
    code = convert_for_evaluation(response or "")
    code_with_test = code + "\n\n" + "\n".join(benchmark["test_list"])
    result = evaluate_code(
        code,
        code_with_test,
        benchmark["instruction_id_list"],
        timeout=3.0,
    )
    rules = {
        rule: bool(result["each_rule_passed"][rule]["passed"])
        for rule in benchmark["instruction_id_list"]
    }
    return {
        "primary_success": bool(result["overall_result"]),
        "component_results": {
            "code_block_extracted": bool(code),
            "functional_tests_passed": bool(result["test"]),
            "functional_test_result": str(result["test_return"]),
            "all_required_style_rules_passed": bool(result["coding_standards"]),
            "required_style_rule_results": rules,
        },
    }


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        result = score(payload)
    except BaseException as exc:
        result = {
            "primary_success": False,
            "component_results": {
                "evaluator_exception_type": type(exc).__name__,
                "evaluator_exception": str(exc),
            },
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
