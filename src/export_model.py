"""export the registered prod model to a local directory so the API (and its Docker image) load it
by PATH — serving never touches the MLflow registry / sqlite. run: python -m src.export_model
(after src.train has registered churn-base12@prod). re-run whenever the prod model changes.
"""
import shutil
import tempfile
from pathlib import Path

import mlflow

from . import config

DEST = config.REPO_ROOT / "api" / "model"


def export():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"

    # download to a temp dir, then copy the dir that actually holds MLmodel into api/model,
    # so the API can always load a fixed path (api/model/MLmodel) regardless of nesting.
    tmp = Path(tempfile.mkdtemp())
    downloaded = Path(mlflow.artifacts.download_artifacts(uri, dst_path=str(tmp)))
    model_dir = next(downloaded.rglob("MLmodel")).parent

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, DEST)
    shutil.rmtree(tmp, ignore_errors=True)

    assert (DEST / "MLmodel").exists(), "export missing MLmodel"
    print(f"exported {uri} -> {DEST}")
    return DEST


if __name__ == "__main__":
    export()
