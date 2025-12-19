#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tmux_all.py

A portable tmux session launcher driven by a YAML config.

Key goals:
- Start a tmux session and create windows/panes, running commands in each pane.
- Provide a small "mini-language" (directives starting with "-") to express common actions:
  splits, working directory, env, sleeps, compose helpers, ROS helpers, etc.
- Provide day-to-day tmux utilities: list sessions, attach, kill, show help.

Requires:
- tmux >= 3.0 recommended (works with older, but some UI niceties vary)
- libtmux (Python)

Example:
  ./tmux_all.py start -c run_tmux_compose.yaml
  ./tmux_all.py attach -s my_session
  ./tmux_all.py list
  ./tmux_all.py kill -s my_session
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import shlex
import libtmux
import yaml


DEFAULT_CONFIG = "run_tmux_config.yaml"
DEFAULT_SESSION = "tmux_all"


HELP_TEXT = r"""
Common tmux keys (default):
  Prefix: Ctrl-b

  Sessions:
    Ctrl-b d          detach (leave session running)
    tmux ls           list sessions (outside tmux)
    tmux attach -t X  attach session X (outside tmux)

  Windows:
    Ctrl-b c          create window
    Ctrl-b ,          rename window
    Ctrl-b n / p      next / previous window
    Ctrl-b 0..9       select window by number

  Panes:
    Ctrl-b %          split vertically (side-by-side)
    Ctrl-b "          split horizontally (top/bottom)
    Ctrl-b o          next pane
    Ctrl-b x          kill pane
    Ctrl-b z          zoom/unzoom pane
    Ctrl-b { / }      swap pane

  Copy mode:
    Ctrl-b [          enter copy mode (scroll/search)
    q                 quit copy mode

In this launcher, a "help" window can be created with `--with-help`
(or set `help_window: true` in YAML).
"""


# ---------------------------
# Config model + utilities
# ---------------------------

@dataclass
class LauncherConfig:
    session: str = DEFAULT_SESSION
    root_dir: str = "."
    env: Dict[str, str] = field(default_factory=dict)
    tmux: Dict[str, Any] = field(default_factory=dict)
    help_window: bool = True
    windows: Dict[str, List[str]] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: Path) -> "LauncherConfig":
        data = yaml.safe_load(path.read_text()) or {}
        # Backward compat: older configs used top-level "window"
        windows = data.get("windows") or data.get("window") or {}
        return LauncherConfig(
            session=data.get("session", DEFAULT_SESSION),
            root_dir=data.get("root_dir", data.get("root", ".")),
            env=data.get("env", {}) or {},
            tmux=data.get("tmux", {}) or {},
            help_window=bool(data.get("help_window", True)),
            windows=windows,
        )


def _expand_vars(s: str, env: Dict[str, str]) -> str:
    """! @brief Expand ${VARS} in a string.

    Supports a small subset of shell / Compose style expansion:
      - ${VAR}
      - ${VAR:-default}

    Resolution order:
      1) @p env mapping (config + CLI overrides)
      2) process environment (os.environ)
      3) default (if provided) or empty string

    @param s Input string.
    @param env Variable overrides.
    @return Expanded string.
    """
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

    def repl(m: re.Match) -> str:
        key = m.group(1)
        default = m.group(2)
        if key in env:
            return env[key]
        if key in os.environ:
            return os.environ[key]
        return default if default is not None else ""

    return pattern.sub(repl, s)


def _merge_env(config_env: Dict[str, str], cli_env: List[str]) -> Dict[str, str]:
    """! @brief Merge env defaults from config and CLI.

    CLI entries are KEY=VALUE strings.

    @param config_env Environment mapping from YAML.
    @param cli_env List of KEY=VALUE strings from CLI.
    @return Merged environment dictionary.
    """
    merged = dict(config_env or {})
    for item in cli_env or []:
        if "=" not in item:
            raise ValueError(f"Invalid --env item '{item}'. Expected KEY=VALUE.")
        k, v = item.split("=", 1)
        merged[k] = v
    return merged


def _resolve_root_dir(config_path: Path, root_dir: str) -> Path:
    """! @brief Resolve a root directory relative to the YAML config.

    If @p root_dir is relative, it is interpreted relative to the YAML file location.

    @param config_path Path to YAML config.
    @param root_dir Root directory string (absolute or relative).
    @return Absolute, resolved Path.
    """
    p = Path(root_dir)
    if not p.is_absolute():
        # root_dir is relative to config file location
        p = (config_path.parent / p).resolve()
    return p


# ---------------------------
# tmux wrappers
# ---------------------------

def tmux_server() -> libtmux.Server:
    """! @brief Create a libtmux Server object.

    @return A connected libtmux.Server instance bound to the default tmux socket.
    """
    return libtmux.Server()


def find_session(server: libtmux.Server, name: str) -> Optional[libtmux.Session]:
    """! @brief Find a tmux session by name.

    Uses QueryList filtering (recommended in recent libtmux versions) instead of
    deprecated dict-like access on Session/Window/Pane objects.

    @param server libtmux server.
    @param name Session name.
    @return Session if found, otherwise None.
    """

    ql = server.sessions.filter(session_name=name)
    if len(ql) > 1:
        # Shouldn't happen in normal tmux usage, but warn and pick first.
        from warnings import warn

        warn(f"Multiple sessions named '{name}'\n{ql}")
    return ql[0] if ql else None


def list_sessions(server: libtmux.Server) -> List[str]:
    """! @brief List session names on the server.

    @param server libtmux server.
    @return List of session names.
    """
    return [s.session_name for s in server.sessions]


def ensure_session(server: libtmux.Server, name: str, *, recreate: bool = False) -> libtmux.Session:
    """! @brief Ensure a session exists (create if missing).

    @param server libtmux server.
    @param name Session name.
    @param recreate If True, kill any existing session with the same name first.
    @return Session object.
    """

    existing = find_session(server, name)
    if existing and recreate:
        # Newer libtmux exposes Session.kill() for `tmux kill-session`.
        existing.kill()
        existing = None
    if existing:
        return existing

    # Create detached session.
    # Note: libtmux is pre-1.0 and args may evolve; session_name/attach are stable.
    return server.new_session(session_name=name, attach=False)


def apply_tmux_options(session: libtmux.Session, options: Dict[str, Any]) -> None:
    """! @brief Apply tmux options to a session.

    Values come from the YAML config under `tmux:`.

    Example:
    @code
    tmux:
      mouse: "on"
      history-limit: 200000
    @endcode

    @param session Target tmux session.
    @param options Mapping of option name to value.
    """
    if not options:
        return
    for k, v in options.items():
        try:
            session.set_option(str(k), str(v))
        except Exception:
            # Don't fail the whole startup for an option; log to stderr.
            print(f"[warn] Failed to set tmux option {k}={v}", file=sys.stderr)


def _pane_bootstrap(pane: libtmux.Pane, root_dir: Path, env: Dict[str, str]) -> None:
    """! @brief Bootstrap a pane with working directory and environment.

    Sends commands into the pane to:
      1) cd into @p root_dir
      2) export all variables from @p env

    @param pane Target tmux pane.
    @param root_dir Root directory for commands.
    @param env Environment to export into the pane.
    """
    # cd first (avoid expensive python sleeps; do everything in-pane)
    pane.send_keys(f"cd {sh_quote(str(root_dir))}", enter=True)
    # Export default env
    for k, v in env.items():
        pane.send_keys(f"export {k}={sh_quote(v)}", enter=True)


def sh_quote(s: str) -> str:
    """! @brief Quote a string for POSIX-ish shell consumption.

    @param s Input string.
    @return Shell-safe quoted string.
    """
    if s == "":
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_./:@%+-]+", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ---------------------------
# Mini-language directives
# ---------------------------

@dataclass
class ExecContext:
    root_dir: Path
    env: Dict[str, str]
    compose_file: Optional[Path] = None
    compose_project: Optional[str] = None
    ros1_setup: Optional[str] = None
    ros2_setup: Optional[str] = None
    workspace_setup: Optional[str] = None  # e.g., install/setup.bash
    # The "current pane" where following commands are executed
    pane: Optional[libtmux.Pane] = None


def directive_help() -> str:
    return textwrap.dedent(
        """
        Mini-language directives (items starting with "-"):

          Layout / panes:
            -v                 split vertically (side-by-side), continue in new pane
            -h                 split horizontally (top/bottom), continue in new pane
            -layout=<name>      set window layout (e.g., even-vertical, tiled)
            -select-pane=<n>    select pane index n in current window (0-based)

          Flow:
            -sleep=<sec>        sleep in the pane (non-blocking for the launcher)
            -gsleep=<sec>       sleep globally (blocks launcher before continuing)
            -host=<cmd>         run command on host (blocking); good for builds
            -print=<msg>        print message on host (launcher output)
            -echo=<msg>         echo message in the pane

          Working directory:
            -cd=<path>          cd inside current pane (relative to root_dir unless absolute)
            -root=<path>        change root_dir for subsequent commands in this window

          Environment:
            -env=K=V            export K=V in current pane (and update context for expansions)
            -unset=K            unset K in current pane (and context)
            -envfile=<path>     load KEY=VALUE lines from a file (relative to root_dir unless absolute)

          ROS helpers:
            -ros1               source /opt/ros/$ROS_DISTRO/setup.bash (ROS 1 style)
            -ros2               source /opt/ros/$ROS_DISTRO/setup.bash (ROS 2 style)
            -ros1-setup=<path>  source a specific ROS1 setup.bash
            -ros2-setup=<path>  source a specific ROS2 setup.bash
            -ws=<path>          source workspace setup.bash (relative to root_dir unless absolute)

          Docker Compose helpers (shortcuts):
            -compose-file=<p>   set COMPOSE_FILE for this window (relative to root_dir unless absolute)
            -compose-proj=<p>   set Compose project name (-p)
            -up=<svc>           docker compose up <svc> (foreground, logs in pane)
            -upd=<svc>          docker compose up -d <svc>
            -build              docker compose build
            -logs=<svc>         docker compose logs -f --tail=200 <svc>
            -exec=<svc>:<cmd>   docker compose exec <svc> <cmd>
            -run=<svc>:<cmd>    docker compose run --rm <svc> <cmd>

        Any non-directive item is executed literally in the pane.
        """
    ).strip()


def _abs_or_under(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


def _load_envfile(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(str(path))
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def _compose_cmd(ctx: ExecContext, *parts: str) -> str:
    cmd = ["docker", "compose"]
    if ctx.compose_project:
        cmd += ["-p", ctx.compose_project]
    if ctx.compose_file:
        cmd += ["-f", str(ctx.compose_file)]
    cmd += list(parts)
    return " ".join(sh_quote(p) for p in cmd)


def apply_directive(ctx: ExecContext, window: libtmux.Window, token: str) -> None:
    """! @brief Apply a single mini-language directive.

    Directives are config list items starting with "-".

    @param ctx Execution context (mutable: env/root_dir/pane).
    @param window Current tmux window.
    @param token Directive token (e.g. "-v", "-sleep=2", "-up=task").
    @throws ValueError on unknown/invalid directive.
    """
    assert ctx.pane is not None, "Pane must be set before executing directives"

    if token == "-v":
        ctx.pane = ctx.pane.split_window(vertical=True, attach=False)
        _pane_bootstrap(ctx.pane, ctx.root_dir, ctx.env)
        return

    if token == "-h":
        ctx.pane = ctx.pane.split_window(vertical=False, attach=False)
        _pane_bootstrap(ctx.pane, ctx.root_dir, ctx.env)
        return

    if token.startswith("-layout="):
        layout = token.split("=", 1)[1].strip()
        window.select_layout(layout)
        return

    if token.startswith("-select-pane="):
        idx = int(token.split("=", 1)[1].strip())
        panes = window.panes
        if idx < 0 or idx >= len(panes):
            raise ValueError(f"Pane index out of range: {idx} (have {len(panes)})")
        ctx.pane = panes[idx]
        return

    if token.startswith("-sleep="):
        sec = token.split("=", 1)[1].strip()
        # Run sleep inside pane so launcher doesn't block
        ctx.pane.send_keys(f"sleep {sh_quote(sec)}", enter=True)
        return

    if token.startswith("-gsleep=") or token.startswith("-global-sleep="):
        sec = token.split("=", 1)[1].strip()
        # Block the launcher: wait before continuing with next directives/windows
        try:
            time.sleep(float(sec))
        except ValueError:
            raise ValueError(f"Invalid global sleep seconds: {sec!r}")
        return

    if token.startswith("-host="):
        cmd = token.split("=", 1)[1].strip()
        # Run command on the host (outside tmux) and wait for completion.
        env = os.environ.copy()
        env.update({k: v for k, v in ctx.env.items() if v is not None})
        result = subprocess.run(cmd, shell=True, cwd=str(ctx.root_dir), env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Host command failed ({result.returncode}): {cmd}")
        return

    if token.startswith("-print="):
        msg = token.split("=", 1)[1]
        print(msg)
        return

    if token.startswith("-echo="):
        msg = token.split("=", 1)[1]
        ctx.pane.send_keys(f"echo {sh_quote(msg)}", enter=True)
        return

    if token.startswith("-cd="):
        p = token.split("=", 1)[1].strip()
        target = _abs_or_under(ctx.root_dir, p)
        ctx.pane.send_keys(f"cd {sh_quote(str(target))}", enter=True)
        return

    if token.startswith("-root="):
        p = token.split("=", 1)[1].strip()
        ctx.root_dir = _abs_or_under(ctx.root_dir, p)
        ctx.pane.send_keys(f"cd {sh_quote(str(ctx.root_dir))}", enter=True)
        return

    if token.startswith("-envfile="):
        p = token.split("=", 1)[1].strip()
        fp = _abs_or_under(ctx.root_dir, p)
        new_env = _load_envfile(fp)
        for k, v in new_env.items():
            ctx.env[k] = v
            ctx.pane.send_keys(f"export {k}={sh_quote(v)}", enter=True)
        return

    if token.startswith("-env="):
        kv = token.split("=", 1)[1]
        if "=" not in kv:
            raise ValueError(f"Invalid -env directive: {token}. Expected -env=K=V.")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        ctx.env[k] = v
        ctx.pane.send_keys(f"export {k}={sh_quote(v)}", enter=True)
        return

    if token.startswith("-unset="):
        k = token.split("=", 1)[1].strip()
        ctx.env.pop(k, None)
        ctx.pane.send_keys(f"unset {sh_quote(k)}", enter=True)
        return

    # ROS helpers
    if token == "-ros1":
        setup = ctx.ros1_setup or "/opt/ros/${ROS_DISTRO}/setup.bash"
        setup = _expand_vars(setup, ctx.env)
        ctx.pane.send_keys(f"source {sh_quote(setup)}", enter=True)
        return

    if token == "-ros2":
        setup = ctx.ros2_setup or "/opt/ros/${ROS_DISTRO}/setup.bash"
        setup = _expand_vars(setup, ctx.env)
        ctx.pane.send_keys(f"source {sh_quote(setup)}", enter=True)
        return

    if token.startswith("-ros1-setup="):
        ctx.ros1_setup = token.split("=", 1)[1].strip()
        return

    if token.startswith("-ros2-setup="):
        ctx.ros2_setup = token.split("=", 1)[1].strip()
        return

    if token.startswith("-ws="):
        p = token.split("=", 1)[1].strip()
        ws = _abs_or_under(ctx.root_dir, p)
        ctx.workspace_setup = str(ws)
        ctx.pane.send_keys(f"source {sh_quote(str(ws))}", enter=True)
        return

    # Compose helpers
    if token.startswith("-compose-file="):
        p = token.split("=", 1)[1].strip()
        ctx.compose_file = _abs_or_under(ctx.root_dir, p)
        return

    if token.startswith("-compose-proj="):
        ctx.compose_project = token.split("=", 1)[1].strip()
        return

    if token == "-build":
        ctx.pane.send_keys(_compose_cmd(ctx, "build"), enter=True)
        return

    if token.startswith("-up="):
        svc = token.split("=", 1)[1].strip()
        ctx.pane.send_keys(_compose_cmd(ctx, "up", "--no-deps", "--remove-orphans", svc), enter=True)
        return

    if token.startswith("-upd="):
        svc = token.split("=", 1)[1].strip()
        ctx.pane.send_keys(_compose_cmd(ctx, "up", "-d", "--no-deps", "--remove-orphans", svc), enter=True)
        return

    if token.startswith("-logs="):
        svc = token.split("=", 1)[1].strip()
        ctx.pane.send_keys(_compose_cmd(ctx, "logs", "-f", "--tail=200", svc), enter=True)
        return

    if token.startswith("-exec="):
        rest = token.split("=", 1)[1]
        if ":" not in rest:
            raise ValueError(f"Invalid -exec directive: {token}. Expected -exec=svc:cmd.")
        svc, cmd = rest.split(":", 1)
        ctx.pane.send_keys(_compose_cmd(ctx, "exec", svc.strip(), "bash", "-lc", cmd.strip()), enter=True)
        return

    if token.startswith("-run="):
        rest = token.split("=", 1)[1]
        if ":" not in rest:
            raise ValueError(f"Invalid -run directive: {token}. Expected -run=svc:cmd.")
        svc, cmd = rest.split(":", 1)
        ctx.pane.send_keys(_compose_cmd(ctx, "run", "--rm", "--service-ports", svc.strip(), "bash", "-lc", cmd.strip()), enter=True)
        return

    raise ValueError(f"Unknown directive: {token}")


def run_windows(session: libtmux.Session, config_path: Path, cfg: LauncherConfig, merged_env: Dict[str, str],
                root_dir: Path, compose_file: Optional[Path], compose_project: Optional[str]) -> None:
    """! @brief Create windows/panes and run configured commands.

    @param session Target tmux session.
    @param config_path Path to YAML configuration.
    @param cfg Parsed configuration.
    @param merged_env Default environment (YAML + CLI overrides).
    @param root_dir Root directory for commands.
    @param compose_file Optional docker compose file to use for shortcut directives.
    @param compose_project Optional compose project name.
    """
    windows = cfg.windows or {}
    if not windows:
        raise ValueError("Config has no windows. Expected 'windows:' (or legacy 'window:') mapping.")

    # first window already exists in a new session; reuse it
    first = True

    for window_name, items in windows.items():
        if first:
            win = session.windows[0]
            win.rename_window(window_name)
            first = False
        else:
            win = session.new_window(window_name=window_name, attach=False)

        # Prepare context and bootstrap pane 0
        pane0 = win.panes[0]
        ctx = ExecContext(
            root_dir=root_dir,
            env=dict(merged_env),  # per-window mutable view
            compose_file=compose_file,
            compose_project=compose_project,
            pane=pane0,
        )
        _pane_bootstrap(ctx.pane, ctx.root_dir, ctx.env)

        # Expand + execute items
        for raw in items or []:
            if raw is None:
                continue
            if not isinstance(raw, str):
                raise TypeError(f"Window '{window_name}' has a non-string item: {raw!r}")

            token = _expand_vars(raw, ctx.env)

            if token.startswith("-"):
                apply_directive(ctx, win, token)
            else:
                ctx.pane.send_keys(token, enter=True)

        # Default layout: don't force if user already set it via -layout=
        try:
            if getattr(win, "window_layout", None):
                # If user didn't specify, make it readable.
                win.select_layout("even-vertical")
        except Exception:
            pass


def create_help_window(session: libtmux.Session) -> None:
    """! @brief Create a "help" window explaining tmux keys and directives.

    @param session Target tmux session.
    """
    win = session.new_window(window_name="help", attach=False)
    pane = win.panes[0]

    # Render help content in one shot for a clean, readable pane (avoid echo-spam).
    help_text = HELP_TEXT.strip("\n")
    directives = directive_help().strip("\n")

    py = textwrap.dedent(r'''
    import os
    import shutil
    import textwrap

    # ANSI styling (safe in most terminals; falls back to plain if NO_COLOR is set)
    NO_COLOR = bool(os.environ.get("NO_COLOR"))
    if NO_COLOR:
        BOLD = DIM = RESET = CYAN = YELLOW = ""
    else:
        BOLD = "\033[1m"
        DIM = "\033[2m"
        RESET = "\033[0m"
        CYAN = "\033[36m"
        YELLOW = "\033[33m"

    width = shutil.get_terminal_size((100, 20)).columns
    width = max(60, min(width, 120))

    def hr(ch="─"):
        print(ch * width)

    def center(s):
        s = s.strip("\n")
        pad = max(0, (width - len(s)) // 2)
        return (" " * pad) + s

    def heading(title):
        print()
        hr("═")
        print(center(f"{BOLD}{title}{RESET}"))
        hr("═")

    def boxed(body):
        lines = body.splitlines()
        inner = width - 4
        top = "┌" + ("─" * (width - 2)) + "┐"
        bot = "└" + ("─" * (width - 2)) + "┘"
        print(top)
        for ln in lines:
            for wrapped in textwrap.wrap(ln, inner, replace_whitespace=False, drop_whitespace=False) or [""]:
                wrapped = wrapped.rstrip("\n")
                print("│ " + wrapped.ljust(inner) + " │")
        print(bot)

    HELP = r"""
    TMUX BASICS
      • Prefix key:         Ctrl-b
      • Help:               Ctrl-b ?
      • Detach:             Ctrl-b d
      • List sessions:      tmux ls
      • Attach:             tmux a -t <name>

    PANES & WINDOWS
      • Split vertical:     Ctrl-b %
      • Split horizontal:   Ctrl-b "
      • Next pane:          Ctrl-b o
      • Close pane:         Ctrl-b x
      • New window:         Ctrl-b c
      • Next window:        Ctrl-b n / p
      • Rename window:      Ctrl-b ,

    SCROLL / COPY MODE
      • Enter copy mode:    Ctrl-b [
      • Search:             /   (in copy mode)
      • Quit copy mode:     q
    """.strip("\n")

    DIRS = r"""
    MINI-LANGUAGE DIRECTIVES (in YAML command lists)
      -v / -h                 Split pane (vertical / horizontal)
      -sleep=SEC              Sleep inside pane (non-blocking for launcher)
      -gsleep=SEC             Global sleep (blocks launcher before continuing)
      -cd=PATH                cd PATH inside pane
      -env=K=V                export K=V inside pane
      -unset=K                unset K inside pane
      -envfile=PATH           source an env file inside pane (bash)
      -host=CMD               Run CMD on host (blocking) before continuing

    DOCKER COMPOSE SHORTCUTS
      -build                  docker compose build
      -up=SVC                 docker compose up --no-deps SVC
      -upd=SVC                docker compose up -d --no-deps SVC
      -logs=SVC               docker compose logs -f --tail=200 SVC
      -exec=SVC:CMD           docker compose exec SVC bash -lc CMD
      -run=SVC:CMD            docker compose run --rm SVC bash -lc CMD
    """.strip("\n")

    heading("TMUX QUICK HELP")
    boxed(HELP)

    heading("LAUNCHER DIRECTIVES")
    boxed(DIRS)

    print()
    print(f"{DIM}Tip:{RESET} In tmux press {BOLD}Ctrl-b ?{RESET} to see key bindings (depends on your tmux config).")
''').strip("\n")

    cmd = "clear; python3 - <<'PY'\n" + py + "\nPY"
    pane.send_keys("bash -lc " + shlex.quote(cmd), enter=True)

    # Keep it readable
    try:
        win.select_layout("even-vertical")
    except Exception:
        pass


# ---------------------------
# CLI
# ---------------------------

def cmd_start(args: argparse.Namespace) -> int:
    """! @brief CLI handler: start.

    Loads YAML config, applies CLI overrides, ensures tmux session, optionally
    creates help window, then creates windows/panes and runs commands.

    @param args Parsed argparse args.
    @return Process exit code.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        # try relative to script directory
        candidate = (Path(__file__).resolve().parent / args.config).resolve()
        if candidate.exists():
            config_path = candidate
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = LauncherConfig.from_yaml(config_path)

    # CLI overrides
    session_name = args.session or cfg.session
    root_dir = _resolve_root_dir(config_path, args.root_dir or cfg.root_dir)
    merged_env = _merge_env(cfg.env, args.env)

    # allow env to be expanded in root_dir itself
    root_dir = Path(_expand_vars(str(root_dir), merged_env)).resolve()

    compose_file = None
    if args.compose_file:
        compose_file = _abs_or_under(root_dir, _expand_vars(args.compose_file, merged_env))
    elif "COMPOSE_FILE" in merged_env:
        compose_file = _abs_or_under(root_dir, _expand_vars(merged_env["COMPOSE_FILE"], merged_env))
    else:
        # convenience: auto-discover docker-compose.yml in root_dir
        candidate = root_dir / "docker-compose.yml"
        if candidate.exists():
            compose_file = candidate

    compose_project = args.compose_project or merged_env.get("COMPOSE_PROJECT_NAME")

    server = tmux_server()
    session = ensure_session(server, session_name, recreate=args.recreate)

    apply_tmux_options(session, cfg.tmux)

    if args.with_help or cfg.help_window:
        # Avoid duplicating help window if restarting into existing session without recreate
        existing_help = [w for w in session.windows if getattr(w, "window_name", None) == "help"]
        if not existing_help:
            create_help_window(session)

    run_windows(session, config_path, cfg, merged_env, root_dir, compose_file, compose_project)

    print(
        f"tmux session '{session_name}' started.\n"
        f"Attach with: tmux attach-session -t {session_name}\n"
        f"List sessions: tmux ls\n"
        f"Kill session:  {Path(sys.argv[0]).name} kill {session_name}\n"
    )
    return 0


def cmd_attach(args: argparse.Namespace) -> int:
    """! @brief CLI handler: attach.

    Uses exec to replace the current process with `tmux attach-session`.

    @param args Parsed argparse args.
    @return Process exit code.
    """
    server = tmux_server()
    name = args.session
    sess = find_session(server, name)
    if not sess:
        print(f"No such session: {name}", file=sys.stderr)
        return 2
    # Use tmux client attach (more reliable than libtmux attach for terminals)
    os.execvp("tmux", ["tmux", "attach-session", "-t", name])


def cmd_list(args: argparse.Namespace) -> int:
    """! @brief CLI handler: list.

    @param args Parsed argparse args.
    @return Process exit code.
    """
    server = tmux_server()
    sessions = list_sessions(server)
    if not sessions:
        print("(no tmux sessions)")
        return 0
    for s in sessions:
        print(s)
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    """! @brief CLI handler: kill.

    @param args Parsed argparse args.
    @return Process exit code.
    """
    server = tmux_server()
    name = args.session
    sess = find_session(server, name)
    if not sess:
        print(f"No such session: {name}", file=sys.stderr)
        return 2
    # Newer libtmux uses Session.kill() (tmux kill-session)
    sess.kill()
    print(f"Killed session: {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Launch and manage tmux sessions from a YAML config.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("start", help="Start (or reuse) a tmux session from config.")
    ps.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help="YAML config path (default: %(default)s).",
    )
    ps.add_argument("--session", help="Session name override (overrides YAML).")
    ps.add_argument("--root-dir", help="Root directory override (overrides YAML).")
    ps.add_argument("--compose-file", help="Compose file path override (relative to root-dir unless absolute).")
    ps.add_argument("--compose-project", help="Compose project name override (-p).")
    ps.add_argument("--env", action="append", default=[], help="Default env KEY=VALUE (reusable by all windows). Repeatable.")
    ps.add_argument("--recreate", action="store_true", help="If session exists, kill and recreate it.")
    ps.add_argument("--with-help", action="store_true", help="Always create a help window (even if YAML disables it).")
    ps.set_defaults(func=cmd_start)

    pa = sub.add_parser("attach", help="Attach to a tmux session.")
    pa.add_argument("session", help="Session name.")
    pa.set_defaults(func=cmd_attach)

    pl = sub.add_parser("list", help="List tmux sessions.")
    pl.set_defaults(func=cmd_list)

    pk = sub.add_parser("kill", help="Kill a tmux session.")
    pk.add_argument("session", help="Session name.")
    pk.set_defaults(func=cmd_kill)

    ph = sub.add_parser("help", help="Print tmux + mini-language help.")
    ph.set_defaults(func=lambda _a: (print(HELP_TEXT.strip()), print(), print(directive_help()), 0)[-1])

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
