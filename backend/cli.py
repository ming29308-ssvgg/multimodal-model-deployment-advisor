import argparse
import json
import sys

from backend.agents.model_selector.graph import run_recommendation


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="自动检测硬件，并根据自然语言任务推荐模型与部署方式。"
    )
    parser.add_argument("task", help="要完成的任务，例如：把会议视频转写并生成摘要")
    parser.add_argument(
        "--quality",
        choices=["speed", "balanced", "quality"],
        default="balanced",
    )
    parser.add_argument("--max-candidates", type=int, default=3)
    args = parser.parse_args()

    result = run_recommendation(
        args.task,
        {
            "quality": args.quality,
            "max_candidates_per_stage": args.max_candidates,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

