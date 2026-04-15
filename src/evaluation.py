"""sed_eval wrapper for per-file event-based evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import CLASSES, COLLARS


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _to_sed_eval_events(events: List[Tuple[float, float, str]], filename: str) -> list:
    out = []
    for start, end, cls in events:
        out.append(
            {
                "event_label": cls,
                "onset": float(start),
                "offset": float(end),
                "file": filename,
            }
        )
    return out


def eval_events_per_file(
    pred_events: List[Tuple[float, float, str]],
    ref_events: List[Tuple[float, float, str]],
    filename: str = "audio.wav",
) -> Dict[str, float]:
    """Evaluate a single file with class-specific collars.

    Returns a dict with ``f1_b``, ``f1_mb``, ``f1_h`` and ``f1_macro``.
    """
    import sed_eval
    import dcase_util

    results: Dict[str, float] = {}
    f1s: List[float] = []
    for cls in CLASSES:
        ref = [e for e in ref_events if e[2] == cls]
        pred = [e for e in pred_events if e[2] == cls]
        ref_list = dcase_util.containers.MetaDataContainer(_to_sed_eval_events(ref, filename))
        pred_list = dcase_util.containers.MetaDataContainer(_to_sed_eval_events(pred, filename))
        metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=[cls],
            t_collar=COLLARS[cls]["t_collar"],
            percentage_of_length=COLLARS[cls]["percentage_of_length"],
            empty_system_output_handling="zero_score",
        )
        metrics.evaluate(reference_event_list=ref_list, estimated_event_list=pred_list)
        f1 = metrics.results_overall_metrics().get("f_measure", {}).get("f_measure", 0.0)
        if f1 is None or (isinstance(f1, float) and np.isnan(f1)):
            f1 = 0.0
        results[f"f1_{cls}"] = float(f1)
        f1s.append(float(f1))
    results["f1_macro"] = safe_mean(f1s)
    return results
