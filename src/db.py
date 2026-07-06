"""one place the duckdb connection is opened, so every module (features, train, score) gets
the same connection settings instead of copy-pasting the connect call."""
import os
import duckdb
from . import config


def connect(path=None, read_only=False):
    # transactions/user_logs are VIEWs over relative CSV paths ('../data/*.csv'), built when the
    # notebooks ran from notebooks/. VERIFIED EMPIRICALLY: duckdb re-resolves a view's path
    # against the process's CURRENT cwd on every query (not the cwd at connect-time, not the
    # path the view was created from) — so cwd must stay at notebooks/ for as long as this
    # connection is queried, not just during connect(). we chdir and deliberately do NOT
    # restore it; the caller's script ends or it explicitly chdirs back when done with `con`.
    db_path = str(path or config.DB_PATH)
    nb_dir = config.REPO_ROOT / "notebooks"
    if nb_dir.is_dir():
        os.chdir(nb_dir)
    con = duckdb.connect(db_path, read_only=read_only)
    con.execute("SET enable_progress_bar = false")  # match the notebooks: quiet the duckdb widgets
    return con
