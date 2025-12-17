#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run(cmd, check=True, cwd=None, log=None):
    if log:
        log(f"$ {shlex.join(cmd)}")
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if log and (p.stdout.strip() or p.stderr.strip()):
        if p.stdout.strip():
            log(p.stdout.rstrip())
        if p.stderr.strip():
            log(p.stderr.rstrip())
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)
    return p

def is_git_repo():
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], check=True)
        return True
    except Exception:
        return False

def git_root():
    return run(["git", "rev-parse", "--show-toplevel"], check=True).stdout.strip()

def current_branch():
    b = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], check=True).stdout.strip()
    if b == "HEAD":
        raise RuntimeError("当前处于 detached HEAD 状态，无法自动 push。请先 checkout 到某个分支。")
    return b

def ensure_remote(remote, log):
    p = run(["git", "remote"], check=True, log=log)
    remotes = set([x.strip() for x in p.stdout.splitlines() if x.strip()])
    if remote not in remotes:
        raise RuntimeError(f"找不到 remote '{remote}'。已有 remotes: {sorted(remotes)}")
    run(["git", "remote", "get-url", remote], check=True, log=log)

def have_git_lfs():
    try:
        run(["git", "lfs", "version"], check=True)
        return True
    except Exception:
        return False

def ensure_lfs_local(log):
    run(["git", "lfs", "install", "--local"], check=True, log=log)

def ensure_gitignore_defaults(log):
    gi = Path(".gitignore")
    defaults = [
        ".DS_Store",
        "._*",
        ".Spotlight-V100/",
        ".Trashes/",
    ]
    existing = set()
    if gi.exists():
        existing = set(gi.read_text(encoding="utf-8", errors="ignore").splitlines())
    to_add = [x for x in defaults if x not in existing]
    if to_add:
        with gi.open("a", encoding="utf-8") as f:
            f.write("\n# --- obsidian_lfs_sync defaults ---\n")
            for x in to_add:
                f.write(x + "\n")
        run(["git", "add", ".gitignore"], check=False, log=log)

def lfs_track(pattern, log):
    run(["git", "lfs", "track", pattern], check=True, log=log)

def parse_status_paths():
    out = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        stdout=subprocess.PIPE,
        check=True
    ).stdout
    parts = out.split(b"\x00")
    paths = []
    i = 0
    while i < len(parts) and parts[i]:
        entry = parts[i].decode("utf-8", errors="replace")
        xy = entry[:2]
        path = entry[3:]
        if xy and xy[0] in ("R", "C"):
            if i + 1 < len(parts) and parts[i+1]:
                new_path = parts[i+1].decode("utf-8", errors="replace")
                paths.append(new_path)
                i += 2
                continue
        paths.append(path)
        i += 1
    # 去重保持顺序
    return list(dict.fromkeys([p for p in paths if p]))

def file_size_bytes(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None

def human(n):
    if n is None:
        return "N/A"
    x = float(n)
    for u in ["B","KB","MB","GB","TB"]:
        if x < 1024 or u == "TB":
            return f"{x:.2f}{u}"
        x /= 1024.0

def is_under_dir(path, dirname):
    p = Path(path)
    try:
        return dirname in p.parts and p.parts[0] == dirname
    except Exception:
        return False

def ensure_upstream_or_set(remote, branch, log):
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace"
    )
    if r.returncode == 0 and r.stdout.strip():
        return False
    run(["git", "push", "-u", remote, branch], check=True, log=log)
    return True

def push_with_retry(remote, branch, retries, log):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            run(["git", "push", remote, branch], check=True, log=log)
            run(["git", "lfs", "push", remote, branch], check=False, log=log)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            err = (e.stderr or "") + "\n" + (e.output or "")
            err_lower = err.lower()
            log(f"[push attempt {attempt}/{retries}] failed.")
            if ("non-fast-forward" in err_lower) or ("fetch first" in err_lower) or ("rejected" in err_lower):
                log("Detected non-fast-forward/rejected. Running: git pull --rebase ...")
                run(["git", "pull", "--rebase", remote, branch], check=True, log=log)
            else:
                sleep_s = min(2 * attempt, 6)
                log(f"Retrying after {sleep_s}s ...")
                time.sleep(sleep_s)
    raise last_err if last_err else RuntimeError("push failed with unknown error")

# -------- lock auto-fix (NEW) --------
def lock_age_seconds(lock_path: Path):
    try:
        return time.time() - lock_path.stat().st_mtime
    except OSError:
        return None

def lsof_pids(lock_path: Path):
    """
    macOS 通常有 lsof。用 `lsof -t` 拿 PID：
      - 有占用：return [pid...]
      - 无占用：return []
      - 没有 lsof：return None
    """
    try:
        p = subprocess.run(
            ["lsof", "-t", "--", str(lock_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError:
        return None
    if p.returncode == 0 and p.stdout.strip():
        pids = []
        for line in p.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids
    return []

def auto_fix_index_lock(lock_path: Path, log, stale_seconds: int):
    """
    安全自动修复策略：
    1) 如果 lock 被进程占用 -> 不删，报错并退出
    2) 未被占用且 age > stale_seconds -> 备份并删除
    3) 未被占用但 age 太新 -> 认为可能是刚启动的 git 操作，报错退出
    """
    if not lock_path.exists():
        return

    age = lock_age_seconds(lock_path)
    pids = lsof_pids(lock_path)

    if pids is not None and len(pids) > 0:
        raise RuntimeError(f"检测到 {lock_path} 正被进程占用（PID={pids}）。请关闭相关 git 客户端/操作后重试。")

    if age is None:
        raise RuntimeError(f"检测到 {lock_path} 但无法读取时间戳，建议手动检查是否有 git 进程在跑。")

    if age < stale_seconds:
        raise RuntimeError(
            f"检测到 {lock_path}（age={int(age)}s）较新，可能仍有 git 操作在进行。"
            f"请稍后重试，或关闭 GitHub Desktop/VSCode 等再试。"
        )

    # stale & not held -> backup then remove
    backup_dir = Path(".git") / "obsidian_lfs_sync_logs" / "lock_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"index.lock.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        lock_path.replace(backup)
        log(f"⚠️ 发现陈旧锁文件，已自动备份并移除：{lock_path} -> {backup}")
    except Exception:
        # fallback: copy then unlink
        data = lock_path.read_bytes()
        backup.write_bytes(data)
        lock_path.unlink(missing_ok=True)
        log(f"⚠️ 发现陈旧锁文件，已自动备份并删除：{lock_path} -> {backup}")

# ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Obsidian vault: LFS-first auto commit+push with rollback on failure.")
    ap.add_argument("--remote", default="origin_github")   # ✅你的默认 remote
    ap.add_argument("--require-branch", default="main")    # ✅保护：必须在 main
    ap.add_argument("--threshold-mb", type=float, default=20.0, help="大文件阈值（非 Book 目录的 PDF 触发自动 LFS track）。默认 20MB")
    ap.add_argument("--book-dir", default="Book", help="默认把该目录下的 PDF 全部走 LFS。默认 Book")
    ap.add_argument("--book-track", default="pdf", help="Book 目录下要走 LFS 的扩展名（逗号分隔）。默认 pdf")
    ap.add_argument("--other-track-mode", choices=["path", "ext"], default="path",
                    help="非 Book 的“大 PDF”自动 track 方式：path=精确到文件路径；ext=按 *.pdf 扩展名。默认 path")
    ap.add_argument("--msg", default=None, help="提交信息。不填则自动时间戳。")
    ap.add_argument("--retries", type=int, default=3, help="push 重试次数。默认 3")

    # ✅ 新增：自动修复 lock 的控制项
    ap.add_argument("--auto-fix-lock", action="store_true", default=True, help="自动修复陈旧的 .git/index.lock（默认开启）")
    ap.add_argument("--no-auto-fix-lock", dest="auto_fix_lock", action="store_false", help="关闭自动修复 lock")
    ap.add_argument("--lock-stale-seconds", type=int, default=180, help="lock 超过多少秒视为陈旧可自动清理（默认 180s）")

    ap.add_argument("--dry-run", action="store_true", help="只打印计划与检测结果，不做任何修改。")
    args = ap.parse_args()

    if not is_git_repo():
        print("❌ 当前目录不在 git 仓库内。请 cd 到 obsidian vault 仓库根目录再运行。", file=sys.stderr)
        sys.exit(1)

    os.chdir(git_root())

    log_dir = Path(".git") / "obsidian_lfs_sync_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(s):
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"[{now_str()}] {s}\n")
        print(s)

    # ✅ 自动修复 index.lock（安全策略）
    lock = Path(".git") / "index.lock"
    if lock.exists():
        if args.auto_fix_lock:
            try:
                auto_fix_index_lock(lock, log, args.lock_stale_seconds)
            except Exception as e:
                log(f"❌ {e}")
                sys.exit(2)
        else:
            log(f"❌ 检测到 {lock}，说明另一个 git 操作未结束。请关闭其它 git 客户端或稍后再试。")
            sys.exit(2)

    try:
        ensure_remote(args.remote, log)

        if not have_git_lfs():
            log("❌ 未检测到 git-lfs。请先安装：brew install git-lfs")
            sys.exit(3)

        branch = current_branch()
        if args.require_branch and branch != args.require_branch:
            raise RuntimeError(f"当前分支是 {branch}，但脚本要求在 {args.require_branch} 上运行。请先 git checkout {args.require_branch}。")

        if not args.dry_run:
            ensure_gitignore_defaults(log)

        changed_paths = parse_status_paths()
        if not changed_paths:
            log("✅ 工作区没有变更，无需提交/推送。")
            return

        threshold_bytes = int(args.threshold_mb * 1024 * 1024)
        book_exts = [e.strip().lower().lstrip(".") for e in args.book_track.split(",") if e.strip()]
        book_dir = args.book_dir.strip("/")

        large_other_pdfs = []
        for p in changed_paths:
            if not os.path.exists(p):
                continue
            ext = Path(p).suffix.lower().lstrip(".")
            size = file_size_bytes(p)
            if size is None:
                continue
            if ext == "pdf" and size >= threshold_bytes and not is_under_dir(p, book_dir):
                large_other_pdfs.append((p, size))

        log("=== Plan ===")
        log(f"Repo: {os.getcwd()}")
        log(f"Branch: {branch}, Remote: {args.remote}")
        log(f"Book-dir: {book_dir}, Book LFS exts: {book_exts}")
        log(f"Other large PDF threshold: {args.threshold_mb}MB, mode: {args.other_track_mode}")
        log(f"Detected changed files: {len(changed_paths)}")

        if large_other_pdfs:
            log(f"Detected large PDFs outside '{book_dir}': {len(large_other_pdfs)}")
            for p, sz in large_other_pdfs[:10]:
                log(f"  - {p} ({human(sz)})")
            if len(large_other_pdfs) > 10:
                log("  ...")

        if args.dry_run:
            log(f"dry-run: 日志已写入 {log_file}")
            return

        ensure_lfs_local(log)

        # Book：默认全部走 LFS（两条规则覆盖深层）
        for e in book_exts:
            lfs_track(f"{book_dir}/*.{e}", log)
            lfs_track(f"{book_dir}/**/*.{e}", log)

        # 非 Book 的大 PDF：按 path / ext track
        if large_other_pdfs:
            if args.other_track_mode == "ext":
                lfs_track("*.pdf", log)
            else:
                for p, _ in large_other_pdfs:
                    lfs_track(p, log)

        run(["git", "add", ".gitattributes"], check=False, log=log)
        run(["git", "add", "-A"], check=True, log=log)

        msg = args.msg or f"notes: auto sync {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        staged = run(["git", "diff", "--cached", "--name-only"], check=True, log=log)
        if staged.stdout.strip():
            run(["git", "commit", "-m", msg], check=True, log=log)
            log(f"✅ committed: {msg}")
        else:
            log("✅ 没有 staged 变更（无需 commit）。")

        try:
            ensure_upstream_or_set(args.remote, branch, log)
        except subprocess.CalledProcessError:
            pass

        push_with_retry(args.remote, branch, args.retries, log)
        log("✅ push success.")
        log(f"Log saved: {log_file}")

    except Exception as e:
        # 失败：回滚到“未提交状态”
        try:
            log(f"❌ ERROR: {repr(e)}")
            orig = subprocess.run(
                ["git", "rev-parse", "--verify", "ORIG_HEAD"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace"
            )
            target = orig.stdout.strip() if (orig.returncode == 0 and orig.stdout.strip()) else "HEAD@{1}"
            log(f"Rolling back to uncommitted state: git reset --mixed {target}")
            run(["git", "reset", "--mixed", target], check=False, log=log)
            log(f"⚠️ 已回滚为未提交状态。请查看日志定位原因：{log_file}")
        except Exception as e2:
            print(f"❌ 回滚也失败了：{repr(e2)}\n请手动查看日志：{log_file}", file=sys.stderr)
        sys.exit(10)

if __name__ == "__main__":
    main()
