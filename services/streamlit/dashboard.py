"""Streamlit dashboard to explore training artifacts and metrics."""

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "metrics.json"
FPS_PATH = PROJECT_ROOT / "reports/fps_metrics.json"
PLOT_SPECS = [
    (["BoxPR_curve.png", "PR_curve.png"], "Precision–Recall curve"),
    (["confusion_matrix.png", "confusion_matrix_normalized.png"], "Confusion matrix"),
    (["results.png"], "Training metrics overview"),
]


def load_metrics() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        st.error(f"metrics.json not found at {METRICS_PATH}")
        return {}
    with METRICS_PATH.open() as fh:
        return json.load(fh)


def load_fps_metrics() -> pd.DataFrame:
    if not FPS_PATH.exists():
        return pd.DataFrame()
    with FPS_PATH.open() as fh:
        payload = json.load(fh)
    measurements = payload.get("measurements", [])
    if not measurements:
        return pd.DataFrame()
    df = pd.DataFrame(measurements)
    df["resolution"] = df["resolution"].astype(int)
    df = df.sort_values("resolution")
    df["model"] = payload.get("model", "")
    return df


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_run_root(model_path: str) -> Path:
    model_path_resolved = resolve_path(model_path)
    if not model_path_resolved.exists():
        return model_path_resolved
    weights_dir = model_path_resolved.parent
    return weights_dir.parent if weights_dir.name == "weights" else weights_dir


def render_metric_table(section_name: str, section: Dict[str, Any]) -> None:
    st.subheader(f"Metrics – {section_name}")
    table_rows = []
    for label, info in section.items():
        if not isinstance(info, dict):
            continue
        metrics = info.get("metrics")
        if not isinstance(metrics, dict):
            continue
        table_rows.append({
            "model": label,
            "model_path": info.get("model_path", ""),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mAP50": metrics.get("mAP50"),
            "mAP50-95": metrics.get("mAP50-95"),
        })
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows))
    else:
        st.info("No metrics recorded yet.")


def format_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def render_experiment_summary(section_name: str, section: Dict[str, Any]) -> None:
    st.header("Experiment Summary")
    dataset = section.get("dataset", "Unknown dataset")
    notes = section.get("notes")
    st.markdown(f"**Experiment:** `{section_name}`  ")
    st.markdown(f"**Dataset:** `{dataset}`")
    if notes:
        st.info(notes)

    custom_metrics = section.get("custom_model", {}).get("metrics", {})
    baseline_metrics = section.get("pretrained_baseline", {}).get("metrics", {})

    if not custom_metrics:
        st.warning("Custom model metrics are not recorded yet.")
        return

    col1, col2, col3 = st.columns(3)

    def delta_for(key: str) -> Any:
        custom_value = custom_metrics.get(key)
        baseline_value = baseline_metrics.get(key)
        if custom_value is None or baseline_value is None:
            return None
        return f"{custom_value - baseline_value:+.3f}"

    col1.metric(
        "mAP50",
        format_metric(custom_metrics.get("mAP50")),
        delta_for("mAP50"),
    )
    col2.metric(
        "Precision",
        format_metric(custom_metrics.get("precision")),
        delta_for("precision"),
    )
    col3.metric(
        "Recall",
        format_metric(custom_metrics.get("recall")),
        delta_for("recall"),
    )

    st.markdown(
        """
        **How to read this page**

        1. The metrics table compares your fine-tuned weights against the reference model.
        2. The training artifacts surface qualitative diagnostics: the precision–recall curve, the confusion matrix, and Ultralytics' training dashboard.
        3. The line chart beneath them shows how key metrics evolved across epochs.
        4. The benchmark section translates model speed into FPS for different image sizes, helping you pick the right compromise between accuracy and latency.
        """
    )


def render_training_plots(run_root: Path) -> None:
    if not run_root.exists():
        st.warning(f"Run folder not found: {run_root}")
        return

    st.subheader("Training Artifacts")
    cols = st.columns(len(PLOT_SPECS))
    for idx, (candidates, label) in enumerate(PLOT_SPECS):
        plot_path = next(
            (run_root / name for name in candidates if (run_root / name).exists()),
            None,
        )
        with cols[idx]:
            if plot_path is not None:
                st.image(str(plot_path), caption=label)
            else:
                missing_label = ", ".join(candidates)
                st.caption(f"Missing: {label} ({missing_label})")

    results_csv = run_root / "results.csv"
    if results_csv.exists():
        results_df = pd.read_csv(results_csv)
        if "epoch" in results_df.columns:
            plot_cols = [
                col
                for col in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"]
                if col in results_df.columns
            ]
            if plot_cols:
                st.line_chart(results_df.set_index("epoch")[plot_cols])
                st.caption("Metric trend across epochs")
    else:
        st.caption("results.csv not available for detailed metric trends.")


def render_fps_section() -> None:
    fps_df = load_fps_metrics()
    st.header("Inference Benchmark (FPS vs. Resolution)")
    if fps_df.empty:
        st.info("Run `python scripts/benchmark_fps.py --model <path>` to capture FPS metrics.")
        return

    model_name = fps_df["model"].iloc[0] if "model" in fps_df.columns else ""
    if model_name:
        st.caption(f"Model: {model_name}")

    chart_df = fps_df.set_index("resolution")[["fps"]]
    st.bar_chart(chart_df)
    st.dataframe(fps_df[["resolution", "fps", "avg_time_ms", "runs"]])


def main() -> None:
    st.set_page_config(page_title="YOLOv8 Evaluation Dashboard", layout="wide")
    st.title("YOLOv8 Week 3 Evaluation Dashboard")
    st.sidebar.markdown(
        """
        **Navigation**

        - Pick an experiment to explore its metrics.
        - The summary highlights what changed after fine-tuning.
        - Scroll for diagnostic plots and FPS benchmarking.
        """
    )

    metrics = load_metrics()
    if not metrics:
        return

    sections = sorted(metrics.keys())
    selected_section_name = st.sidebar.selectbox("Select experiment", sections)
    selected_section = metrics[selected_section_name]

    render_experiment_summary(selected_section_name, selected_section)
    render_metric_table(selected_section_name, {
        "custom_model": selected_section.get("custom_model", {}),
        "pretrained_baseline": selected_section.get("pretrained_baseline", {}),
    })

    custom_model_path = selected_section.get("custom_model", {}).get("model_path")
    if custom_model_path:
        st.sidebar.write("### Custom Model Run")
        st.sidebar.code(custom_model_path)
        run_root = get_run_root(custom_model_path)
        render_training_plots(run_root)

    with st.expander("Raw metrics.json"):
        st.json(metrics)

    render_fps_section()


if __name__ == "__main__":
    main()
