"""What the installed journal hooks run: ``python -m llmoji.hook <provider>``.

Reads the harness's Stop payload on stdin, turns the turn it fired on
into rows and appends the new ones. It never fails the harness: errors
go to ``~/.llmoji/hook-errors.log`` and the exit status is always 0.
"""

from __future__ import annotations

import json
import sys
import time
import traceback


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        from .capture import append_rows
        from .providers import get_provider

        provider = get_provider(args[0])
        payload = json.loads(sys.stdin.read() or "{}")
        if isinstance(payload, dict):
            append_rows(provider.journal_path, provider.live_rows(payload))
    except Exception:  # noqa: BLE001 — a hook must never break the harness
        _log(args, traceback.format_exc())
    return 0


def _log(args: list[str], tb: str) -> None:
    try:
        from .paths import hook_errors_path

        path = hook_errors_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            f.write(f"{stamp} llmoji.hook {' '.join(args)}\n{tb}\n")
    except OSError:
        pass  # a sandboxed harness may not let us write here either


if __name__ == "__main__":
    sys.exit(main())
