"""Command line entry point:  python -m coolroof <command>

  data    audit + clean the raw CSV, write reports and figures
  train   train/compare/evaluate all models on the cleaned data
  insights  explainability + business insights
  all     data -> train -> insights
  serve   start the FastAPI backend
"""
from __future__ import annotations

import argparse
import sys


def cmd_data() -> None:
    from .data import clean, report

    res = clean.clean(clean.load_raw())
    clean.save_artifacts(res)
    report.run_all(res)
    s = res.summary
    print(f"cleaned {s['n_raw_rows']} rows -> {C_PATH()}")
    print("eligible rows by task:", s["eligible_rows_by_task"])


def C_PATH():
    from . import config as C
    return C.CLEANED_CSV


def cmd_train() -> None:
    from .ml import train
    train.run()


def cmd_insights() -> None:
    from .ml import insights
    insights.run()


def cmd_serve(host: str, port: int) -> None:
    import uvicorn
    uvicorn.run("coolroof.api.main:app", host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="coolroof", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("data", "train", "insights", "all"):
        sub.add_parser(name)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    if a.cmd == "data":
        cmd_data()
    elif a.cmd == "train":
        cmd_train()
    elif a.cmd == "insights":
        cmd_insights()
    elif a.cmd == "all":
        cmd_data(); cmd_train(); cmd_insights()
    elif a.cmd == "serve":
        cmd_serve(a.host, a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
