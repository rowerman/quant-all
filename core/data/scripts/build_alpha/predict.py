"""重放 Alpha 模型预测。"""

from __future__ import annotations

import argparse
import json

from alpha.predict import predict


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    parser = argparse.ArgumentParser(description="重放 Alpha 模型预测")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser


def main() -> None:
    """命令行入口。"""

    args = _parser().parse_args()
    frame = predict(args.model_id, args.start, args.end)
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "rows": int(len(frame)),
                "dates": int(frame["trade_date"].nunique()),
                "start": str(frame["trade_date"].min().date()),
                "end": str(frame["trade_date"].max().date()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
