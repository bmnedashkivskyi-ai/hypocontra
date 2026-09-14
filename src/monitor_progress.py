"""Монітор довготривалих процесів проєкту (мінінг/резолюція/фетч оглядів).

Усі скрипти в src/, що обробляють список елементів (mine_disagreement_from_surveys.py,
resolve_survey_citations.py, check_acl_arxiv_overlap.py, annotate_with_gemma.py,
find_surveys_bulk.py тощо), уже друкують прогрес у форматі "[i/N] ..." -- цей
монітор парсить саме цей формат, тож НЕ вимагає жодних змін у самих скриптах.

Два режими:
  run    -- запускає команду сам, показує живий прогрес + heartbeat, detект
            "зависання" (немає нового рядка довше --stall-timeout секунд).
  attach -- підключається до вже запущеного процесу: читає лог-файл (напр.
            вивід фонової bash-задачі) як `tail -f` і перевіряє живість PID.

Приклади:
  .venv/bin/python src/monitor_progress.py run -- .venv/bin/python src/mine_disagreement_from_surveys.py
  .venv/bin/python src/monitor_progress.py attach --log /tmp/.../task.output --pid 12345
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time

PROGRESS_PATTERN = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")
HEARTBEAT_INTERVAL = 30.0
DEFAULT_STALL_TIMEOUT = 120.0


class ProgressState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current: int | None = None
        self.total: int | None = None
        self.last_line = ""
        self.last_update_ts = time.monotonic()
        self.start_ts = time.monotonic()
        self.stall_announced = False
        self.finished = False
        self.exit_code: int | None = None

    def update(self, line: str) -> None:
        with self.lock:
            self.last_line = line.strip()
            self.last_update_ts = time.monotonic()
            self.stall_announced = False
            m = PROGRESS_PATTERN.search(line)
            if m:
                self.current, self.total = int(m.group(1)), int(m.group(2))

    def snapshot(self) -> tuple[int | None, int | None, str, float]:
        with self.lock:
            return self.current, self.total, self.last_line, time.monotonic() - self.last_update_ts


def format_status(state: ProgressState) -> str:
    current, total, last_line, idle = state.snapshot()
    elapsed = time.monotonic() - state.start_ts
    if current is not None and total is not None and total > 0:
        pct = 100.0 * current / total
        rate = current / elapsed * 60 if elapsed > 0 else 0.0
        remaining = total - current
        eta_min = remaining / rate if rate > 0 else float("inf")
        eta_str = f"~{eta_min:.0f}хв" if eta_min != float("inf") else "?"
        return f"{current}/{total} ({pct:.1f}%) | темп: {rate:.1f}/хв | ETA: {eta_str} | остання дія: {idle:.0f}с тому"
    return f"(прогрес ще не визначено) | остання дія: {idle:.0f}с тому | \"{last_line[:60]}\""


def watchdog(state: ProgressState, stall_timeout: float, pid: int | None, stop_event: threading.Event) -> None:
    """Перевіряє зависання часто (POLL_INTERVAL), але друкує heartbeat-статус
    рідше (HEARTBEAT_INTERVAL) -- баг першої версії: обидва були на 30с,
    через що коротке зависання (менше за heartbeat-інтервал) не ловилось
    до наступного тіка. Перевірено прямим тестом (dummy_stall.py, 8с пауза
    при --stall-timeout 3 -- спрацювало лише після цього виправлення."""
    POLL_INTERVAL = 2.0
    last_heartbeat = time.monotonic()

    while not stop_event.wait(POLL_INTERVAL):
        if state.finished:
            return
        _, _, _, idle = state.snapshot()
        alive = True
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True

        if idle >= stall_timeout:
            with state.lock:
                already = state.stall_announced
                state.stall_announced = True
            if not already:
                status = "живий, але мовчить" if alive else "ПРОЦЕС ЗАВЕРШИВСЯ, останній рядок не був фінальним"
                print(f"\n[monitor] ⚠ МОЖЛИВЕ ЗАВИСАННЯ -- немає нового рядка {idle:.0f}с (поріг {stall_timeout:.0f}с). "
                      f"Процес: {status}. Стан: {format_status(state)}\n", file=sys.stderr, flush=True)
        elif time.monotonic() - last_heartbeat >= HEARTBEAT_INTERVAL:
            last_heartbeat = time.monotonic()
            print(f"[monitor] {format_status(state)}", flush=True)


def run_mode(args: argparse.Namespace) -> int:
    cmd = args.command
    print(f"[monitor] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    state = ProgressState()
    stop_event = threading.Event()
    wd = threading.Thread(target=watchdog, args=(state, args.stall_timeout, proc.pid, stop_event), daemon=True)
    wd.start()

    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        print(line, end="", flush=True)
        state.update(line)

    proc.wait()
    state.finished = True
    stop_event.set()
    wd.join(timeout=2)

    elapsed = time.monotonic() - state.start_ts
    print(f"\n[monitor] Процес завершився з кодом {proc.returncode} за {elapsed / 60:.1f}хв. "
          f"Фінальний стан: {format_status(state)}", flush=True)
    return proc.returncode or 0


def attach_mode(args: argparse.Namespace) -> int:
    log_path = args.log
    pid = args.pid
    print(f"[monitor] attaching to {log_path} (pid={pid})", flush=True)

    state = ProgressState()
    stop_event = threading.Event()
    wd = threading.Thread(target=watchdog, args=(state, args.stall_timeout, pid, stop_event), daemon=True)
    wd.start()

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        try:
            while True:
                line = f.readline()
                if line:
                    print(line, end="", flush=True)
                    state.update(line)
                    continue
                if pid is not None:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        print(f"\n[monitor] PID {pid} більше не існує -- процес завершився.", flush=True)
                        break
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

    state.finished = True
    stop_event.set()
    wd.join(timeout=2)
    print(f"[monitor] Фінальний стан: {format_status(state)}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_run = sub.add_parser("run", help="Запустити команду й моніторити її вивід.")
    p_run.add_argument("command", nargs=argparse.REMAINDER, help="Команда для запуску (після --).")
    p_run.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT)

    p_attach = sub.add_parser("attach", help="Підключитись до вже запущеного процесу через його лог-файл.")
    p_attach.add_argument("--log", required=True, help="Шлях до файлу, куди пише процес (tail -f).")
    p_attach.add_argument("--pid", type=int, default=None, help="PID процесу для перевірки живості.")
    p_attach.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT)

    args = parser.parse_args()
    if args.mode == "run":
        cmd = args.command
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            parser.error("Потрібна команда після 'run --'.")
        args.command = cmd
        sys.exit(run_mode(args))
    else:
        sys.exit(attach_mode(args))


if __name__ == "__main__":
    main()
