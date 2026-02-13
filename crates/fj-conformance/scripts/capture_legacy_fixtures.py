#!/usr/bin/env python3
"""Capture transform fixtures from the legacy JAX oracle.

Usage:
  python crates/fj-conformance/scripts/capture_legacy_fixtures.py \
      --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
      --output /data/projects/frankenjax/crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json

Default behavior:
- try to run true legacy JAX capture
- if JAX import fails, fallback to deterministic analytical capture

Use `--strict` to disable fallback and fail hard on JAX import errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_jax(legacy_root: Path):
    sys.path.insert(0, str(legacy_root))
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    return jax, jnp


@dataclass
class Case:
    case_id: str
    family: str
    mode: str
    program: str
    transforms: list[str]
    args: list[dict[str, Any]]
    expected: list[dict[str, Any]]
    atol: float
    rtol: float


def fixture_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"kind": "scalar_i64", "value": int(value)}
    if isinstance(value, int):
        return {"kind": "scalar_i64", "value": value}
    if isinstance(value, float):
        return {"kind": "scalar_f64", "value": value}
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, int) for item in value):
            return {"kind": "vector_i64", "values": [int(item) for item in value]}
        if all(isinstance(item, (int, float)) for item in value):
            return {"kind": "vector_f64", "values": [float(item) for item in value]}
        raise ValueError("list/tuple fixture values must be numeric")

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(value)
        if arr.ndim == 0:
            if np.issubdtype(arr.dtype, np.integer):
                return {"kind": "scalar_i64", "value": int(arr.item())}
            return {"kind": "scalar_f64", "value": float(arr.item())}
        if arr.ndim == 1:
            if np.issubdtype(arr.dtype, np.integer):
                return {"kind": "vector_i64", "values": [int(x) for x in arr.tolist()]}
            return {"kind": "vector_f64", "values": [float(x) for x in arr.tolist()]}
        raise ValueError(f"Unsupported array rank for fixture capture: {arr.ndim}")
    except ModuleNotFoundError as err:
        raise ValueError(
            "fixture_value received an unsupported object and numpy is unavailable"
        ) from err


def build_cases_with_oracle(jax, jnp) -> list[Case]:
    def add2(x, y):
        return x + y

    def square(x):
        return x * x

    def square_plus_linear(x):
        return x * x + 2.0 * x

    def add_one(x):
        return x + 1

    return [
        Case(
            case_id="jit_add_scalar",
            family="jit",
            mode="strict",
            program="add2",
            transforms=["jit"],
            args=[fixture_value(2), fixture_value(5)],
            expected=[fixture_value(jax.jit(add2)(2, 5))],
            atol=1e-6,
            rtol=1e-6,
        ),
        Case(
            case_id="grad_square_scalar",
            family="grad",
            mode="strict",
            program="square",
            transforms=["grad"],
            args=[fixture_value(3.0)],
            expected=[fixture_value(jax.grad(square)(3.0))],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="grad_square_plus_linear_scalar",
            family="grad",
            mode="strict",
            program="square_plus_linear",
            transforms=["grad"],
            args=[fixture_value(3.0)],
            expected=[fixture_value(jax.grad(square_plus_linear)(3.0))],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="vmap_add_one_vector",
            family="vmap",
            mode="strict",
            program="add_one",
            transforms=["vmap"],
            args=[fixture_value(jnp.array([1, 2, 3]))],
            expected=[fixture_value(jax.vmap(add_one)(jnp.array([1, 2, 3])))],
            atol=1e-6,
            rtol=1e-6,
        ),
        Case(
            case_id="vmap_grad_square_vector",
            family="vmap",
            mode="strict",
            program="square",
            transforms=["vmap", "grad"],
            args=[fixture_value(jnp.array([1.0, 2.0, 3.0]))],
            expected=[
                fixture_value(jax.vmap(jax.grad(square))(jnp.array([1.0, 2.0, 3.0])))
            ],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="jit_grad_square_plus_linear",
            family="jit",
            mode="strict",
            program="square_plus_linear",
            transforms=["jit", "grad"],
            args=[fixture_value(3.0)],
            expected=[fixture_value(jax.jit(jax.grad(square_plus_linear))(3.0))],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="grad_jit_square_plus_linear",
            family="grad",
            mode="strict",
            program="square_plus_linear",
            transforms=["grad", "jit"],
            args=[fixture_value(3.0)],
            expected=[fixture_value(jax.grad(jax.jit(square_plus_linear))(3.0))],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="jit_vmap_add_one_vector",
            family="jit",
            mode="strict",
            program="add_one",
            transforms=["jit", "vmap"],
            args=[fixture_value(jnp.array([1, 2, 3]))],
            expected=[fixture_value(jax.jit(jax.vmap(add_one))(jnp.array([1, 2, 3])))],
            atol=1e-6,
            rtol=1e-6,
        ),
    ]


def build_cases_fallback() -> list[Case]:
    return [
        Case(
            case_id="jit_add_scalar",
            family="jit",
            mode="strict",
            program="add2",
            transforms=["jit"],
            args=[fixture_value(2), fixture_value(5)],
            expected=[fixture_value(7)],
            atol=1e-6,
            rtol=1e-6,
        ),
        Case(
            case_id="grad_square_scalar",
            family="grad",
            mode="strict",
            program="square",
            transforms=["grad"],
            args=[fixture_value(3.0)],
            expected=[fixture_value(6.0)],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="grad_square_plus_linear_scalar",
            family="grad",
            mode="strict",
            program="square_plus_linear",
            transforms=["grad"],
            args=[fixture_value(3.0)],
            expected=[fixture_value(8.0)],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="vmap_add_one_vector",
            family="vmap",
            mode="strict",
            program="add_one",
            transforms=["vmap"],
            args=[fixture_value([1, 2, 3])],
            expected=[fixture_value([2, 3, 4])],
            atol=1e-6,
            rtol=1e-6,
        ),
        Case(
            case_id="vmap_grad_square_vector",
            family="vmap",
            mode="strict",
            program="square",
            transforms=["vmap", "grad"],
            args=[fixture_value([1.0, 2.0, 3.0])],
            expected=[fixture_value([2.0, 4.0, 6.0])],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="jit_grad_square_plus_linear",
            family="jit",
            mode="strict",
            program="square_plus_linear",
            transforms=["jit", "grad"],
            args=[fixture_value(3.0)],
            expected=[fixture_value(8.0)],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="grad_jit_square_plus_linear",
            family="grad",
            mode="strict",
            program="square_plus_linear",
            transforms=["grad", "jit"],
            args=[fixture_value(3.0)],
            expected=[fixture_value(8.0)],
            atol=1e-3,
            rtol=1e-3,
        ),
        Case(
            case_id="jit_vmap_add_one_vector",
            family="jit",
            mode="strict",
            program="add_one",
            transforms=["jit", "vmap"],
            args=[fixture_value([1, 2, 3])],
            expected=[fixture_value([2, 3, 4])],
            atol=1e-6,
            rtol=1e-6,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail if legacy JAX capture cannot run; do not fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.skip_existing and args.output.exists():
        print(f"skip-existing set and output exists: {args.output}")
        return 0

    legacy_root = args.legacy_root
    if not legacy_root.exists():
        print(f"legacy root does not exist: {legacy_root}", file=sys.stderr)
        return 2

    capture_mode = "legacy_jax"
    try:
        jax, jnp = _import_jax(legacy_root)
        cases = build_cases_with_oracle(jax, jnp)
    except Exception as exc:
        if args.strict:
            print(
                "Failed to import/execute JAX from legacy root under --strict mode. "
                "Ensure jax + jaxlib are installed and compatible.",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return 3

        capture_mode = "fallback_analytical"
        cases = build_cases_fallback()

    bundle = {
        "schema_version": "frankenjax.transform-fixtures.v1",
        "generated_by": "legacy_jax_capture_script",
        "generated_at_unix_ms": int(time.time() * 1000),
        "oracle_root": str(legacy_root),
        "capture_mode": capture_mode,
        "strict_capture": bool(args.strict),
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family,
                "mode": case.mode,
                "program": case.program,
                "transforms": case.transforms,
                "args": case.args,
                "expected": case.expected,
                "atol": case.atol,
                "rtol": case.rtol,
            }
            for case in cases
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.output} (capture_mode={capture_mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
