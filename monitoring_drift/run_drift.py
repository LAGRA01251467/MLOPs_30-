import os
import json
import argparse
import numpy as np
import pandas as pd
import yaml
import mlflow
from typing import Dict
from datetime import datetime

from metrics import regression_metrics, compare_metrics
from plot_utils import ensure_dir, plot_feature_hist_pair, plot_error_bars

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_validation(path_csv: str, target_col: str):
    df = pd.read_csv(path_csv)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def simulate_drift(X: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    Xd = X.copy(deep=True)

    # mean shifts
    for col, shift in (cfg.get("mean_shifts") or {}).items():
        if col in Xd.columns:
            Xd[col] = Xd[col].astype(float) + float(shift)

    # zero out features
    for col in (cfg.get("zero_out_features") or []):
        if col in Xd.columns:
            Xd[col] = 0.0

    # gaussian noise
    std_fraction = (cfg.get("noise") or {}).get("std_fraction", 0.0)
    if std_fraction > 0:
        numeric_cols = Xd.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = Xd[col].std()
            noise = np.random.normal(
                loc=0.0,
                scale=std_fraction * (std if std > 0 else 1.0),
                size=len(Xd)
            )
            Xd[col] = Xd[col].astype(float) + noise

    # sample fraction
    frac = float(cfg.get("sample_fraction", 1.0))
    if 0 < frac < 1.0:
        Xd = Xd.sample(frac=frac, random_state=42).reset_index(drop=True)

    return Xd

def get_model(mlflow_cfg: Dict):
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    model_uri = mlflow_cfg["model_uri"]
    return mlflow.pyfunc.load_model(model_uri)

def main(config_path: str):
    cfg = load_config(config_path)
    outdir = cfg["outputs_dir"]
    ensure_dir(outdir)

    # reproducibilidad
    np.random.seed(42)

    # 1) cargar base (validación)
    X_base, y_base = load_validation(cfg["validation_csv"], cfg["target_col"])

    # 2) simular drift
    X_drift = simulate_drift(X_base, cfg["drift"])

    # 3) cargar modelo desde MLflow
    model = get_model(cfg["mlflow"])

    # 4) predicciones y métricas
    y_pred_base = model.predict(X_base)
    y_pred_drift = model.predict(X_drift)

    base_m = regression_metrics(y_base, y_pred_base)

    if len(X_drift) != len(X_base):
        y_base_aligned = y_base.iloc[X_drift.index].reset_index(drop=True)
    else:
        y_base_aligned = y_base.reset_index(drop=True)

    drift_m = regression_metrics(y_base_aligned, y_pred_drift)
    deltas = compare_metrics(base_m, drift_m)

    # 5) alertas
    th = cfg["thresholds"]
    alerts = {
        "mae_delta_alert": deltas["delta_mae"] > th["mae_delta_abs"],
        "rmse_delta_alert": deltas["delta_rmse"] > th["rmse_delta_abs"],
        "r2_drop_alert": deltas["drop_r2"] > th["r2_drop_abs"]
    }
    any_alert = any(alerts.values())
    #   PROPUESTA — SNAPSHOT AUDITABLE DEL ESTADO DEL DRIFT 
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "mean_shifts_applied": cfg["drift"].get("mean_shifts", {}),
        "zero_out_features": cfg["drift"].get("zero_out_features", []),
        "noise_std_fraction": cfg["drift"].get("noise", {}).get("std_fraction", 0),
        "sample_fraction": cfg["drift"].get("sample_fraction", 1.0),
        "thresholds": cfg["thresholds"],
        "alerts": alerts,
        "any_alert": any_alert
    }

    snapshot_path = os.path.join(outdir, "drift_snapshot.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=4)

    # 6) artefactos locales
    X_drift_out = X_drift.copy()
    X_drift_out["y_true"] = y_base_aligned.values
    X_drift_out["y_pred_base"] = np.array(y_pred_base[:len(X_drift_out)])
    X_drift_out["y_pred_drift"] = np.array(y_pred_drift)
    X_drift_out.to_csv(os.path.join(outdir, "drift_dataset_predictions.csv"), index=False)

    with open(os.path.join(outdir, "metrics_base.json"), "w") as f:
        json.dump(base_m, f, indent=2)
    with open(os.path.join(outdir, "metrics_drift.json"), "w") as f:
        json.dump(drift_m, f, indent=2)
    with open(os.path.join(outdir, "metrics_deltas.json"), "w") as f:
        json.dump(deltas, f, indent=2)
    with open(os.path.join(outdir, "alerts.json"), "w") as f:
        json.dump({"alerts": alerts, "any_alert": any_alert}, f, indent=2)

    # 7) gráficos
    feat_altered = list((cfg["drift"].get("mean_shifts") or {}).keys()) \
                 + list((cfg["drift"].get("zero_out_features") or []))

    plot_feature_hist_pair(
        X_base.reset_index(drop=True),
        X_drift.reset_index(drop=True),
        features=feat_altered,
        outdir=os.path.join(outdir, "plots", "features")
    )

    plot_error_bars(
        base_m, drift_m,
        os.path.join(outdir, "plots", "error_comparison.png")
    )

    # 8) logging en MLflow
    mlflow.set_experiment("absenteeism_drift_monitoring")

    with mlflow.start_run(run_name="drift_simulation"):

        # log params
        mlflow.log_params({
            "mean_shifts": str(cfg["drift"].get("mean_shifts")),
            "zero_out_features": str(cfg["drift"].get("zero_out_features")),
            "noise_std_fraction": cfg["drift"].get("noise", {}).get("std_fraction", 0),
            "sample_fraction": cfg["drift"].get("sample_fraction", 1.0),
        })

        # log metrics
        mlflow.log_metrics({
            "base_mae": base_m["mae"],
            "base_rmse": base_m["rmse"],
            "base_r2": base_m["r2"],
            "drift_mae": drift_m["mae"],
            "drift_rmse": drift_m["rmse"],
            "drift_r2": drift_m["r2"],
            "delta_mae": deltas["delta_mae"],
            "delta_rmse": deltas["delta_rmse"],
            "drop_r2": deltas["drop_r2"]
        })

        #   PROPUESTA — TAGS DE DRIFT EN MLFLOW
        mlflow.set_tags({
            "drift_any_alert": str(any_alert),
            "drift_mae_delta_alert": str(alerts["mae_delta_alert"]),
            "drift_rmse_delta_alert": str(alerts["rmse_delta_alert"]),
            "drift_r2_drop_alert": str(alerts["r2_drop_alert"])
        })

        # snapshot como artifact
        mlflow.log_artifact(snapshot_path)

        # log artifacts completos
        mlflow.log_artifacts(outdir)

    # 9) salida consola
    print("=== BASE ===", base_m)
    print("=== DRIFT ===", drift_m)
    print("=== DELTAS ===", deltas)
    print("=== ALERTS ===", alerts, "ANY_ALERT:", any_alert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="monitoring_drift/drift_config.yaml")
    args = parser.parse_args()
    main(args.config)

