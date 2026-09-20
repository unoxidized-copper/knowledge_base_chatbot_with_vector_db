from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "scripts/build_index.py"])
    run([sys.executable, "scripts/evaluate_retrieval.py"])
    run([sys.executable, "scripts/ask.py", "প্রফুল্ল শ্বশুরবাড়ি যেতে চেয়েছিল কেন?"])
    run([sys.executable, "scripts/ask.py", "প্রফুল্ল কি ঢাকা বিশ্ববিদ্যালয়ে পড়েছিল?"])


if __name__ == "__main__":
    main()
