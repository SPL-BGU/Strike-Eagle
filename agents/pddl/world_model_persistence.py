"""
Save / load PDDLAgent world model + KB to a JSON sidecar.

JSON holds physics hyperparams, sample states, LR coef/intercept/R², and a
human-readable summary. Sklearn / M5 model objects live in a base64 pickle
blob under ``_pickle_b64`` so ENHSP injection works after reload.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import pickle
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


class StubM5:
    """Minimal stand-in for M5 tree model used in inject.
    
    Defined here so pickle can find it when loading saved world models
    that were created by scripts/verify_world_model_kb.py.
    """

    def predict(self, X):
        return [0.0] * len(X)


class _BackwardsCompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps StubM5 from __main__ to this module."""

    def find_class(self, module: str, name: str):
        # Remap StubM5 from __main__ or verify script to this module
        if name == "StubM5":
            return StubM5
        return super().find_class(module, name)

from agents.pddl.pddl_files.world_model.params import Params

BUNDLE_VERSION = 1
ROLLING_VARS = ("x", "y", "v_x", "v_y")
KB_EVENTS = ("collision", "platform_collision")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _serialize_lr_models_json(lr_models: Optional[dict]) -> Optional[dict]:
    if not isinstance(lr_models, dict):
        return None
    out = {}
    for key, entry in lr_models.items():
        if key == "_feature_flags":
            out[key] = copy.deepcopy(entry)
            continue
        if not isinstance(entry, dict):
            continue
        out[key] = {
            k: entry[k]
            for k in ("r2", "coef", "intercept", "n_features")
            if k in entry
        }
    return out or None


def _strip_event_kb(event: dict) -> dict:
    if not isinstance(event, dict):
        return {}
    out = {
        "states": copy.deepcopy(event.get("states") or []),
        "variables": {},
    }
    for var_name, var_data in (event.get("variables") or {}).items():
        out["variables"][var_name] = {
            "value": copy.deepcopy(var_data.get("value") or []),
        }
    lr_pddl = _serialize_lr_models_json(event.get("lr_models_pddl"))
    if lr_pddl:
        out["lr_models_pddl"] = lr_pddl
    lr_sim = _serialize_lr_models_json(event.get("lr_models"))
    if lr_sim:
        out["lr_models"] = lr_sim
    if "learning_history" in event:
        out["learning_history"] = copy.deepcopy(event["learning_history"])
    return out


def _strip_kb_for_json(kb: dict) -> dict:
    out = {}
    for key, val in (kb or {}).items():
        if key == "trajectories":
            trajectories = []
            for traj in val or []:
                if hasattr(traj, "tolist"):
                    trajectories.append(traj.tolist())
                else:
                    trajectories.append(traj)
            out[key] = trajectories
        elif key in KB_EVENTS:
            out[key] = _strip_event_kb(val)
        elif key.startswith("_"):
            out[key] = copy.deepcopy(val)
    return out


def _extract_kb_models(kb: dict) -> dict:
    models = {}
    for event_name in KB_EVENTS:
        event = (kb or {}).get(event_name) or {}
        event_models = {}
        for var_name, var_data in (event.get("variables") or {}).items():
            model = var_data.get("model")
            mc = var_data.get("model_comparison")
            if model is not None or mc is not None:
                event_models[var_name] = {
                    "model": model,
                    "model_comparison": mc,
                    "loo_cv_history": var_data.get("loo_cv_history"),
                }
        if event_models:
            models[event_name] = event_models
    return models


def _extract_lr_sklearn(kb: dict) -> dict:
    out = {}
    for event_name in KB_EVENTS:
        event = (kb or {}).get(event_name) or {}
        lr_blob = {}
        for lr_key in ("lr_models", "lr_models_pddl"):
            lr_models = event.get(lr_key)
            if not isinstance(lr_models, dict):
                continue
            lr_sklearn = {}
            for var_name, entry in lr_models.items():
                if var_name == "_feature_flags":
                    continue
                if isinstance(entry, dict) and entry.get("model") is not None:
                    lr_sklearn[var_name] = entry["model"]
            if lr_sklearn:
                lr_blob[lr_key] = lr_sklearn
        if lr_blob:
            out[event_name] = lr_blob
    return out


def _build_summary(agent) -> dict:
    kb = agent.kb or {}
    summary = {
        "collision_samples": len((kb.get("collision") or {}).get("states") or []),
        "platform_collision_samples": len((kb.get("platform_collision") or {}).get("states") or []),
        "trajectory_count": len(kb.get("trajectories") or []),
        "kb_force_samples": len(getattr(agent, "kb_force", []) or []),
        "bambirds_scale_n": getattr(agent, "bambirds_scale_n", 0),
    }
    hp = agent.world_model.hyperparams_values if agent.world_model else {}
    summary["gravity"] = float(hp.get(Params.gravity, 85))
    summary["velocity"] = float(hp.get(Params.velocity, 180))
    for event_name in KB_EVENTS:
        pddl_lr = ((kb.get(event_name) or {}).get("lr_models_pddl") or {}).get("v_x") or {}
        if pddl_lr.get("r2") is not None:
            summary[f"{event_name}_pddl_vx_r2"] = float(pddl_lr["r2"])
    return summary


def _pickle_learned_transitions(learned: dict) -> dict:
    """Return a pickle-safe copy of learned_transitions (strip unpicklable models)."""
    from agents.pddl.optimizer import ConstantPredictModel

    out = {}
    for var_name, entry in (learned or {}).items():
        if not isinstance(entry, dict):
            out[var_name] = copy.deepcopy(entry)
            continue
        safe = {}
        for key, val in entry.items():
            if key == "model" and val is not None:
                try:
                    pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)
                    safe[key] = val
                except Exception:
                    poly = entry.get("polynomial")
                    if poly is not None and hasattr(poly, "coef"):
                        safe[key] = ConstantPredictModel(float(poly.coef[0]))
                    continue
            else:
                safe[key] = copy.deepcopy(val)
        out[var_name] = safe
    return out


def _pickle_blob(agent) -> str:
    payload = {
        "kb_variable_models": _extract_kb_models(agent.kb),
        "lr_sklearn": _extract_lr_sklearn(agent.kb),
        "force_lr_model": getattr(agent, "force_lr_model", None),
        "learned_transitions": _pickle_learned_transitions(
            getattr(agent, "learned_transitions", {})
        ),
    }
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(raw).decode("ascii")


def default_save_path(root: str = "saved_models") -> str:
    """Timestamped default path for world-model KB snapshots."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = os.path.abspath(root)
    os.makedirs(parent, exist_ok=True)
    return os.path.join(parent, f"world_model_{ts}.json")


def save_world_model(agent, path: str, reason: str = "") -> str:
    """Persist agent physics + KB to JSON. Returns the path written."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    hp = agent.world_model.hyperparams_values if agent.world_model else {}
    learned_meta = {}
    for var_name, entry in (getattr(agent, "learned_transitions", {}) or {}).items():
        if not isinstance(entry, dict):
            continue
        learned_meta[var_name] = {
            k: entry[k]
            for k in ("string", "initial_value")
            if k in entry
        }

    bundle = {
        "version": BUNDLE_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "hyperparams": {
            "gravity": float(hp.get(Params.gravity, 85)),
            "velocity": float(hp.get(Params.velocity, 180)),
        },
        "bambirds_scale": getattr(agent, "bambirds_scale", None),
        "bambirds_scale_n": getattr(agent, "bambirds_scale_n", 0),
        "force_lr_r2": getattr(agent, "force_lr_r2", None),
        "kb_force": [
            [float(row[0]), float(row[1]), float(row[2])]
            for row in (getattr(agent, "kb_force", []) or [])
        ],
        "summary": _build_summary(agent),
        "kb": _strip_kb_for_json(agent.kb),
        "learned_transitions_meta": learned_meta,
        "_pickle_b64": _pickle_blob(agent),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=_json_default)
    print(f"[WORLD-MODEL] Saved to {path} ({reason or 'checkpoint'})")
    print(
        f"[WORLD-MODEL]   gravity={bundle['hyperparams']['gravity']:.2f} "
        f"velocity={bundle['hyperparams']['velocity']:.2f} "
        f"collision_n={bundle['summary']['collision_samples']} "
        f"platform_n={bundle['summary']['platform_collision_samples']}"
    )
    return path


def _empty_event_kb() -> dict:
    return {
        "states": [],
        "variables": {vn: {"value": [], "model": None} for vn in ROLLING_VARS},
    }


def _merge_pickle_into_kb(kb: dict, pickle_payload: dict) -> None:
    kb_models = pickle_payload.get("kb_variable_models") or {}
    for event_name, var_models in kb_models.items():
        event = kb.setdefault(event_name, _empty_event_kb())
        variables = event.setdefault("variables", {})
        for var_name in ROLLING_VARS:
            variables.setdefault(var_name, {"value": [], "model": None})
        for var_name, blob in var_models.items():
            target = variables.setdefault(var_name, {"value": [], "model": None})
            if blob.get("model") is not None:
                target["model"] = blob["model"]
            if blob.get("model_comparison") is not None:
                target["model_comparison"] = blob["model_comparison"]
            if blob.get("loo_cv_history") is not None:
                target["loo_cv_history"] = blob["loo_cv_history"]

    lr_sklearn = pickle_payload.get("lr_sklearn") or {}
    for event_name, lr_blob in lr_sklearn.items():
        event = kb.setdefault(event_name, _empty_event_kb())
        for lr_key, var_models in lr_blob.items():
            lr_models = event.setdefault(lr_key, {})
            for var_name, model in var_models.items():
                entry = lr_models.setdefault(var_name, {})
                entry["model"] = model


def load_world_model(agent, path: str) -> bool:
    """Load physics + KB from JSON sidecar into agent. Returns True on success."""
    if not path or not os.path.isfile(path):
        print(f"[WORLD-MODEL] Load skipped — file not found: {path}")
        return False

    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    version = bundle.get("version", 0)
    if version != BUNDLE_VERSION:
        print(f"[WORLD-MODEL] WARNING: bundle version {version} != {BUNDLE_VERSION}")

    hp = bundle.get("hyperparams") or {}
    agent.world_model = agent.world_model.__class__({
        Params.gravity: float(hp.get("gravity", 85)),
        Params.velocity: float(hp.get("velocity", 180)),
    })

    kb = copy.deepcopy(bundle.get("kb") or {})
    for event_name in KB_EVENTS:
        event = kb.setdefault(event_name, _empty_event_kb())
        variables = event.setdefault("variables", {})
        for var_name in ROLLING_VARS:
            var_entry = variables.setdefault(var_name, {"value": []})
            var_entry.setdefault("model", None)
            var_entry.setdefault("value", [])

    trajectories = []
    for traj in kb.get("trajectories") or []:
        trajectories.append(np.array(traj) if traj else traj)
    kb["trajectories"] = trajectories

    pickle_b64 = bundle.get("_pickle_b64")
    if pickle_b64:
        # Use custom unpickler to handle StubM5 class remapping
        pickle_bytes = base64.b64decode(pickle_b64.encode("ascii"))
        pickle_payload = _BackwardsCompatibleUnpickler(io.BytesIO(pickle_bytes)).load()
        _merge_pickle_into_kb(kb, pickle_payload)
        agent.force_lr_model = pickle_payload.get("force_lr_model")
        learned = pickle_payload.get("learned_transitions") or {}
        if not hasattr(agent, "learned_transitions") or agent.learned_transitions is None:
            agent.learned_transitions = {}
        for var_name, entry in learned.items():
            agent.learned_transitions[var_name] = entry
    else:
        print("[WORLD-MODEL] WARNING: no _pickle_b64 — M5/sklearn models missing")

    meta = bundle.get("learned_transitions_meta") or {}
    if not hasattr(agent, "learned_transitions") or agent.learned_transitions is None:
        agent.learned_transitions = {}
    for var_name, fields in meta.items():
        if agent.learned_transitions.get(var_name) is None:
            agent.learned_transitions[var_name] = {}
        agent.learned_transitions[var_name].update(fields)

    from agents.pddl.pddl_files.events.learn_events import refresh_platform_pddl_vx_ratio

    platform_event = kb.get("platform_collision")
    if isinstance(platform_event, dict):
        refresh_platform_pddl_vx_ratio(platform_event)

    agent.kb = kb
    agent.world_model.kb = kb
    agent.learned_transition_world_model = agent.world_model

    agent.bambirds_scale = bundle.get("bambirds_scale")
    agent.bambirds_scale_n = int(bundle.get("bambirds_scale_n") or 0)
    agent.force_lr_r2 = bundle.get("force_lr_r2")
    agent.kb_force = [
        (float(row[0]), float(row[1]), float(row[2]))
        for row in (bundle.get("kb_force") or [])
    ]

    summary = bundle.get("summary") or {}
    saved_at = bundle.get("saved_at", "?")
    print(f"[WORLD-MODEL] Loaded from {path} (saved {saved_at})")
    print(
        f"[WORLD-MODEL]   gravity={hp.get('gravity', '?')} "
        f"velocity={hp.get('velocity', '?')} "
        f"collision_n={summary.get('collision_samples', '?')} "
        f"platform_n={summary.get('platform_collision_samples', '?')}"
    )
    return True
