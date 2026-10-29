"""Explicit zero-step serialization proof, then run the frozen99 driver.

MuJoCo's first XML serialization includes redundant mesh content_type=model/stl.
Only that literal omission is allowed. Compiled arrays/options and source assets
must match exactly before the unmodified driver performs its normal gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.local import optimize_ollie_phase_basis as driver
from supermimic.ollie.physics import BOARD_XML, make_scene_spec

DRIVER_SHA = "6f0d8b75f3b5d0bdc5612a6a6f0d2ebf57cb00a60398e680207c866bdae94131"
CONTENT_ATTRIBUTE = ' content_type="model/stl"'


def validate_xml_difference(first, second):
    """Byte-for-byte comparison after one specifically authorized attribute removal."""
    if first.replace(CONTENT_ATTRIBUTE, "") != second:
        raise ValueError("Scene XML changed beyond redundant mesh content_type=model/stl")
    # It must be a mesh attribute. Do not exempt another element's attribute.
    import xml.etree.ElementTree as ET

    for node in ET.fromstring(first).iter():
        if "content_type" in node.attrib and (
            node.tag != "mesh" or node.attrib["content_type"] != "model/stl"
        ):
            raise ValueError("Only redundant STL mesh content_type is authorized")
    return {
        "first_xml_sha256": hashlib.sha256(first.encode()).hexdigest(),
        "second_xml_sha256": hashlib.sha256(second.encode()).hexdigest(),
        "removed_redundant_mesh_content_type_count": first.count(CONTENT_ATTRIBUTE),
        "only_authorized_literal_attribute_difference": True,
    }


def compare_compiled_models(first, second):
    """All model arrays and numeric options, including observation/name arrays."""
    arrays, options, dimensions = {}, {}, {}

    def fingerprint(value):
        value = np.ascontiguousarray(value)
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }

    def compare(name, value, other, mapping):
        a, b = fingerprint(value), fingerprint(other)
        if a != b:
            raise ValueError("Compiled model changed: " + name)
        mapping[name] = a

    for name in dir(first):
        if name.startswith("_"):
            continue
        value = getattr(first, name)
        if isinstance(value, np.ndarray):
            compare(name, value, getattr(second, name), arrays)
        elif isinstance(value, (int, float, bool)):
            compare(name, value, getattr(second, name), dimensions)
    for name in dir(first.opt):
        if name.startswith("_"):
            continue
        value = getattr(first.opt, name)
        if isinstance(value, (np.ndarray, int, float, bool)):
            compare("opt." + name, value, getattr(second.opt, name), options)
    if not arrays or not options:
        raise ValueError("No compiled physical arrays/options were checked")
    serialized = json.dumps(
        {"arrays": arrays, "options": options, "dimensions": dimensions}, sort_keys=True
    ).encode()
    return {
        "all_compiled_model_arrays_bitwise_equal": True,
        "all_numeric_options_bitwise_equal": True,
        "array_count": len(arrays),
        "option_count": len(options),
        "compiled_fingerprint_sha256": hashlib.sha256(serialized).hexdigest(),
        "arrays": arrays,
        "options": options,
        "dimensions": dimensions,
    }


def asset_hashes(spec):
    meshes = {}
    for mesh in spec.meshes:
        if mesh.file:
            path = Path(spec.modelfiledir) / spec.meshdir / mesh.file
            meshes[str(path.resolve())] = driver.protocol.digest(path)
    return {
        "board_xml": {
            "path": str(BOARD_XML.resolve()),
            "sha256": driver.protocol.digest(BOARD_XML),
        },
        "mesh_sha256": meshes,
    }


def serialization_preflight(source, expected_source_sha):
    """No MjData, native integration, optimization, force or saved-state application."""
    if driver.protocol.digest(driver.__file__) != DRIVER_SHA:
        raise ValueError("Frozen99 driver source hash differs")
    if driver.protocol.digest(source) != expected_source_sha:
        raise ValueError("Explicit continuation source bytes differ")
    physical = json.loads((source.parent / "frozen_manifest.json").read_text())["canonical_assets"]
    first = make_scene_spec(timestep=0.0025, foot_deck_friction=None)
    first_assets = asset_hashes(first)
    first_xml = first.to_xml()
    first_model = first.compile()
    second = make_scene_spec(timestep=0.0025, foot_deck_friction=None)
    second_assets = asset_hashes(second)
    second_xml = second.to_xml()
    second_model = second.compile()
    xml = validate_xml_difference(first_xml, second_xml)
    compiled = compare_compiled_models(first_model, second_model)
    expected_assets = {name: physical[name] for name in ("board_xml", "mesh_sha256")}
    if first_assets != second_assets or first_assets != expected_assets:
        raise ValueError("Board or mesh source bytes changed")
    # This is a normal call after the declared serialization warmup. No return
    # values or global functions are patched; the frozen driver repeats it.
    warmed = driver.protocol.model_sources()
    if warmed != physical or warmed["canonical_scene_xml_sha256"] != xml["second_xml_sha256"]:
        raise ValueError("Normal warmed model_sources does not match source manifest")
    return {
        "schema": "supermimic.ollie.explicit_serialization_warmup.v1",
        "pass": True,
        "dynamics_steps": 0,
        "optimizer_calls": 0,
        "mjdata_created": False,
        "explicit_serialization_warmup": True,
        "frozen_driver_sha256": DRIVER_SHA,
        "launcher_sha256": driver.protocol.digest(__file__),
        "source": {"path": str(source.resolve()), "sha256": expected_source_sha},
        "source_scene_xml_sha256": physical["canonical_scene_xml_sha256"],
        "xml": xml,
        "compiled_physics": compiled,
        "assets_before_and_after_exact": first_assets,
        "normal_model_sources_after_warmup_exact_source_manifest": True,
        "no_patch_or_bypassed_driver_gate": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "benchmark", "run"), default="inspect")
    parser.add_argument("--nominal-source", type=Path, required=True)
    parser.add_argument("--nominal-sha256", required=True)
    parser.add_argument("--nominal-audit", type=Path, required=True)
    parser.add_argument("--nominal-audit-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.out_dir.with_name(args.out_dir.name + ".serialization_preflight.json")
    if args.out_dir.exists() or sidecar.exists():
        parser.error("Fresh output directory and serialization sidecar are required")
    proof = {
        "pass": False,
        "dynamics_steps": 0,
        "launcher_sha256": driver.protocol.digest(__file__),
    }
    try:
        proof = serialization_preflight(args.nominal_source, args.nominal_sha256)
    except (RuntimeError, ValueError, KeyboardInterrupt) as error:
        proof["error"] = str(error) or type(error).__name__
        raise
    finally:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "serialization_preflight_pass": True,
                "sidecar": str(sidecar.resolve()),
                "sidecar_sha256": driver.protocol.digest(sidecar),
                "array_count": proof["compiled_physics"]["array_count"],
                "option_count": proof["compiled_physics"]["option_count"],
                "xml": proof["xml"],
            }
        ),
        flush=True,
    )
    # Preserve sys.argv unchanged so all normal frozen driver parsing, source
    # audit, source-model, deadline, baseline and actual-vs-branch gates apply.
    driver.main()


if __name__ == "__main__":
    main()
