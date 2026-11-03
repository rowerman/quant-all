"""生成 P1 purged walk-forward 切分表。"""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from alpha.calendar import load_calendar
from alpha.config import DEFAULT_CONTRACT, MODEL_DATA_DIR
from alpha.logging_conf import get_logger
from alpha.split import SplitConfig, build_folds, validate_label_gap

logger = get_logger("scripts.build_alpha.build_folds")


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="生成 purged walk-forward 切分表")
    parser.add_argument("--start", default=DEFAULT_CONTRACT.dev_start)
    parser.add_argument("--end", default=DEFAULT_CONTRACT.dev_end)
    parser.add_argument(
        "--mode", choices=("expanding", "rolling"), default="expanding"
    )
    parser.add_argument("--window-years", type=int, default=None)
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    calendar = load_calendar(args.start, args.end)
    config = SplitConfig(
        dev_start=args.start,
        dev_end=args.end,
        mode=args.mode,
        train_window_years=args.window_years,
    )
    folds = build_folds(calendar, config)
    for fold in folds:
        validate_label_gap(calendar, fold, horizon=config.horizon)
    frame = pd.DataFrame([fold.to_dict() for fold in folds])
    folder = MODEL_DATA_DIR / "splits"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "walk_forward_v1.csv"
    frame.to_csv(csv_path, index=False)
    manifest = {
        "config": config.__dict__,
        "calendar_sha256": hashlib.sha256(
            "\n".join(str(day.date()) for day in calendar).encode("utf-8")
        ).hexdigest(),
        "folds": [fold.to_dict() for fold in folds],
    }
    (folder / "walk_forward_v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("切分表完成：%d 折，写入 %s", len(folds), csv_path)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
