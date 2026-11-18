"""
Tinshi AI Agent - a local Ollama coding agent (Claude-Code-style UI)

- Live streaming text (and code) as the model generates it
- Modes (Tab / Shift+Tab to cycle): default (asks), auto-accept (no prompts), plan (read-only, then proposes a plan)
- /model opens an arrow-key model picker
- Tools: RUN (cmd), RUN_PS (PowerShell), WRITE_FILE, READ_FILE; output is fed back so it can keep working
- Uses local Ollama; key.txt is optional because local Ollama does not require an API key
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import requests

try:
    import msvcrt  # Windows key handling
except ImportError:
    msvcrt = None

if os.name != "nt":
    import termios
    import tty

# ---------- terminal setup ----------
os.system("")  # enables ANSI colors on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ORANGE = "\033[38;2;215;119;87m"
GRAY = "\033[90m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
PURPLE = "\033[38;2;150;130;230m"
TEAL = "\033[38;2;80;200;180m"
RESET = "\033[0m"


def width():
    return min(shutil.get_terminal_size((100, 20)).columns, 120)


def bullet(text, color=ORANGE):
    print(f"\n{color}⏺{RESET} {text}")


def result(text, max_lines=8, color=GRAY):
    lines = (text or "").rstrip().splitlines() or ["(no output)"]
    for i, l in enumerate(lines[:max_lines]):
        print(f"{color}{'  ⎿  ' if i == 0 else '     '}{l}{RESET}")
    if len(lines) > max_lines:
        print(f"{GRAY}     … +{len(lines) - max_lines} lines{RESET}")


def type_out(s, color=""):
    """Print a chunk smoothly, character by character."""
    if not s:
        return
    delay = min(0.004, 0.12 / len(s))
    for ch in s:
        sys.stdout.write(color + ch.replace("\n", "\n  ") + (RESET if color else ""))
        sys.stdout.flush()
        if delay > 0.0005:
            time.sleep(delay)


class Spinner:
    FRAMES = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"]

    def __init__(self, label="Thinking"):
        self.label, self._stop, self._t = label, threading.Event(), None

    def start(self):
        self._stop.clear()
        self.t0 = time.time()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            f = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r\033[K{ORANGE}{f} {self.label}…{RESET} {GRAY}({int(time.time() - self.t0)}s · ctrl+c to interrupt){RESET}",
                  end="", flush=True)
            i += 1
            time.sleep(0.12)

    def stop(self):
        if self._t:
            self._stop.set()
            self._t.join()
            self._t = None
            print("\r\033[K", end="", flush=True)


# ---------- config ----------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
API_URL = OLLAMA_BASE_URL + "/v1/chat/completions"
MODELS_URL = OLLAMA_BASE_URL + "/api/tags"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(SCRIPT_DIR, "key.txt")
SAVE_DIR = os.path.join(SCRIPT_DIR, "chats")
SKILLS_DIR = os.path.join(SCRIPT_DIR, "skills")
MAX_OUTPUT_CHARS = 6000

SKILL_TEMPLATE = """---
name: {name}
description: One line saying WHEN the AI should use this skill.
---

# {name}

Write the instructions here.
"""


def load_skills():
    skills = {}
    if not os.path.isdir(SKILLS_DIR):
        return skills
    paths = []
    for entry in sorted(os.listdir(SKILLS_DIR)):
        full = os.path.join(SKILLS_DIR, entry)
        if entry.lower().endswith(".md"):
            paths.append(full)
        elif os.path.isfile(os.path.join(full, "SKILL.md")):
            paths.append(os.path.join(full, "SKILL.md"))
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        if name.upper() == "SKILL":
            name = os.path.basename(os.path.dirname(path))
        desc, body = "", content
        m = re.match(r"^---\\s*\\n(.*?)\\n---\\s*\\n?(.*)$", content, re.DOTALL)
        if m:
            body = m.group(2)
            for line in m.group(1).splitlines():
                k, _, v = line.partition(":")
                if k.strip() == "name" and v.strip():
                    name = v.strip()
                elif k.strip() == "description":
                    desc = v.strip()
        skills[name.lower()] = {"name": name, "description": desc or "(no description)",
                                "body": body.strip(), "path": path}
    return skills


def skills_prompt(skills):
    if not skills:
        return ""
    lines = "\\n".join(f"- {s['name']}: {s['description']}" for s in skills.values())
    return ("\\n\\nSKILLS: extra instructions you can load. When a task matches a skill's description, "
            "FIRST reply with one line:\\nLOAD_SKILL: <name>\\nthen follow its instructions.\\n"
            f"Available skills:\\n{lines}\\n")


TOOL_PREFIXES = ("RUN", "WRITE", "READ", "LOAD_SKILL")


def fetch_local_models():
    try:
        response = requests.get(MODELS_URL, timeout=8)
        response.raise_for_status()
        data = response.json().get("models", [])
        return [(m.get("name") or m.get("model"), "Local Ollama")
                for m in data if (m.get("name") or m.get("model"))]
    except Exception as e:
        result(f"Couldn't fetch local Ollama models: {e}", color=RED)
        return []


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
APP_NAME = "Tinshi AI Agent"


MODES = ["default", "auto", "plan"]
MODE_STATUS = {
    "default": f"{GRAY}? for shortcuts · tab to cycle modes{RESET}",
    "auto": f"{PURPLE}⏵⏵ auto-accept on{RESET} {GRAY}(tab to cycle){RESET}",
    "plan": f"{TEAL}⏸ plan mode on{RESET} {GRAY}(tab to cycle){RESET}",
}
MODE_PROMPT = {
    "default": "",
    "auto": "",
    "plan": ("\nYOU ARE IN PLAN MODE: do NOT write files or run commands. You may only use READ_FILE "
             "to explore. Then reply in plain text with a clear numbered plan of what you will do. "
             "Do not execute anything until the user approves."),
}


def load_key():
    # Ollama is local and does not require authentication.
    # key.txt is retained for the requested file-based setup; "ollama" is sufficient.
    if os.path.exists(KEY_FILE):
        try:
            return open(KEY_FILE, "r", encoding="utf-8").read().strip() or "ollama"
        except OSError:
            pass
    return "ollama"


API_KEY = load_key()


# ---------- interactive widgets ----------
def pick(title, options, current=0):
    """Arrow-key menu. options = list of (label, description). Returns index or None."""
    if not msvcrt:
        for i, (l, d) in enumerate(options):
            print(f"  {i + 1}. {l}  {GRAY}{d}{RESET}")
        c = input("  number › ").strip()
        return int(c) - 1 if c.isdigit() and 0 < int(c) <= len(options) else None

    idx = current
    view = min(len(options), 12)  # visible rows (scrolls)
    n = view + 2

    def draw(first=False):
        if not first:
            sys.stdout.write(f"\033[{n}A")
        sys.stdout.write(f"\r\033[K {BOLD}{title}{RESET} {GRAY}(↑/↓ move · enter select · esc cancel · {idx + 1}/{len(options)}){RESET}\n\r\033[K\n")
        top = min(max(0, idx - view // 2), len(options) - view)
        for i in range(top, top + view):
            l, d = options[i]
            sel = i == idx
            mark = f"{ORANGE}❯{RESET}" if sel else " "
            name = f"{ORANGE}{l}{RESET}" if sel else l
            cur = f" {GREEN}✔{RESET}" if i == current else ""
            sys.stdout.write(f"\r\033[K {mark} {i + 1}. {name}{cur}  {GRAY}{d}{RESET}\n")
        sys.stdout.flush()

    draw(True)
    while True:
        k = msvcrt.getwch()
        if k in ("\x00", "\xe0"):
            k2 = msvcrt.getwch()
            if k2 == "H":
                idx = (idx - 1) % len(options)
            elif k2 == "P":
                idx = (idx + 1) % len(options)
            draw()
        elif k.isdigit() and 0 < int(k) <= min(9, len(options)) and len(options) <= 9:
            idx = int(k) - 1
            draw()
            return idx
        elif k == "\r":
            return idx
        elif k in ("\x1b", "\x03"):
            return None


class InputBox:
    """Claude-Code-style prompt: rules above/below, status line, Tab cycles mode, ↑/↓ history."""

    def __init__(self):
        self.history = []

    def status(self, agent):
        return f"  {MODE_STATUS[agent.mode]}   {GRAY}{agent.model}{RESET}"

    def read(self, agent):
        w = width()
        if not msvcrt:
            if not sys.stdin.isatty():
                return input("> ").strip()
            print(f"\n{GRAY}{'─' * w}{RESET}\n{BOLD}>{RESET} ", end="", flush=True)
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            buf = ""
            hi = len(self.history)
            try:
                tty.setraw(fd)
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ("\r", "\n"):
                        break
                    if ch == "\x03":
                        raise KeyboardInterrupt
                    if ch in ("\x7f", "\x08"):
                        buf = buf[:-1]
                    elif ch == "\x1b":
                        seq = sys.stdin.read(2)
                        if seq == "[A" and self.history:
                            hi = max(0, hi - 1); buf = self.history[hi]
                        elif seq == "[B" and self.history:
                            hi = min(len(self.history), hi + 1)
                            buf = self.history[hi] if hi < len(self.history) else ""
                        else:
                            buf = ""
                    elif ch == "\t":
                        agent.mode = MODES[(MODES.index(agent.mode) + 1) % len(MODES)]
                    elif ch.isprintable():
                        buf += ch
                    sys.stdout.write(f"\r\033[K{BOLD}>{RESET} {buf}")
                    sys.stdout.flush()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            print(f"\n{GRAY}{'─' * w}{RESET}\n{self.status(agent)}")
            if buf.strip():
                self.history.append(buf)
            return buf.strip()

        sys.stdout.write(f"\n{GRAY}{'─' * w}{RESET}\n{BOLD}>{RESET} \n{GRAY}{'─' * w}{RESET}\n{self.status(agent)}")
        sys.stdout.write("\033[2A\r")  # back up to the input line
        buf, hi = "", len(self.history)

        def redraw_line():
            sys.stdout.write(f"\r\033[K{BOLD}>{RESET} {buf}")
            sys.stdout.flush()

        def redraw_status():
            sys.stdout.write(f"\033[2B\r\033[K{self.status(agent)}\033[2A")
            redraw_line()

        redraw_line()
        while True:
            k = msvcrt.getwch()
            if k == "\r":
                break
            elif k == "\x03":
                sys.stdout.write("\033[2B\n")
                raise KeyboardInterrupt
            elif k == "\x08":
                buf = buf[:-1]
            elif k == "\x1b":
                buf = ""
            elif k == "\t":
                agent.mode = MODES[(MODES.index(agent.mode) + 1) % len(MODES)]
                redraw_status()
                continue
            elif k in ("\x00", "\xe0"):
                k2 = msvcrt.getwch()
                if k2 == "\x0f":  # shift+tab
                    agent.mode = MODES[(MODES.index(agent.mode) + 1) % len(MODES)]
                    redraw_status()
                    continue
                if k2 == "H" and self.history:
                    hi = max(0, hi - 1)
                    buf = self.history[hi]
                elif k2 == "P" and self.history:
                    hi = min(len(self.history), hi + 1)
                    buf = self.history[hi] if hi < len(self.history) else ""
            elif k == "?" and not buf:
                sys.stdout.write("\033[2B\n")
                show_shortcuts()
                return ""
            elif k.isprintable():
                buf += k
            redraw_line()

        sys.stdout.write("\033[2B\n")
        sys.stdout.flush()
        if buf.strip():
            self.history.append(buf)
        return buf.strip()


def show_shortcuts():
    bullet("Shortcuts")
    rows = [("tab / shift+tab", "cycle mode: default → auto-accept → plan"),
            ("↑ / ↓", "prompt history"), ("esc", "clear input"), ("ctrl+c", "interrupt"),
            ("/model", "pick a model"), ("/mode", "pick a mode"), ("/clear", "clear conversation"),
            ("/cd <path>", "change directory"), ("/run", "run a command yourself"),
            ("/steps <n>", "max auto steps per task"), ("/save /load", "save / load chats"),
            ("/skills", "list your skills"), ("/skill new <name>", "create a skill (opens Notepad)"),
            ("/skill edit <name>", "edit a skill"), ("/<skillname> [task]", "run a skill directly"),
            ("/exit", "quit")]
    for c, d in rows:
        print(f"     {ORANGE}{c:<18}{RESET}{GRAY}{d}{RESET}")


# ---------- agent ----------
class Agent:
    def __init__(self):
        self.model = DEFAULT_MODEL
        self.mode = "default"
        self.max_steps = 20
        self.cwd = os.getcwd()
        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": ""}]

    def save(self, name=None):
        os.makedirs(SAVE_DIR, exist_ok=True)
        name = name or datetime.now().strftime("chat_%Y%m%d_%H%M%S")
        path = os.path.join(SAVE_DIR, name if name.endswith(".json") else name + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "messages": self.messages}, f, indent=2)
        result(f"Saved to {path}")

    def load(self, name):
        path = os.path.join(SAVE_DIR, name if name.endswith(".json") else name + ".json")
        if not os.path.exists(path):
            result(f"No such file: {path}", color=RED)
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.model = data.get("model", self.model)
        self.messages = data["messages"]
        result(f"Loaded {len(self.messages) - 1} messages.")

    # ---------- API (streams live to screen) ----------
    def call_api(self):
        self.skills = load_skills()  # re-read every time so new/edited skills work instantly
        self.messages[0]["content"] = (SYSTEM_PROMPT.format(cwd=self.cwd) + skills_prompt(self.skills)
                                       + MODE_PROMPT[self.mode])
        spinner = Spinner("Thinking")
        spinner.start()
        r = None
        try:
            url, key, model_id = API_URL, API_KEY, self.model
            for attempt in range(3):
                try:
                    r = requests.post(
                        url,
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model_id, "messages": self.messages,
                              "max_tokens": 8000, "temperature": 0.2, "stream": True},
                        timeout=(15, 300),  # 15s to connect, up to 5 min waiting for data
                        stream=True,
                    )
                except requests.Timeout:
                    if attempt == 2:
                        raise
                    spinner.label = f"Server busy, retrying ({attempt + 2}/3)"
                    continue
                if r.status_code in (429, 502, 503, 504) and attempt < 2:
                    spinner.label = f"Server busy ({r.status_code}), retrying ({attempt + 2}/3)"
                    time.sleep(5 * (attempt + 1))
                    continue
                break
        except KeyboardInterrupt:
            spinner.stop()
            print(f"\n{GRAY}  ⎿  Interrupted{RESET}")
            return None
        except (requests.RequestException, RuntimeError) as e:
            spinner.stop()
            bullet(f"{RED}Connection error{RESET}", RED)
            result(str(e), color=RED)
            if isinstance(e, requests.Timeout):
                result("The server is overloaded. Try again in a bit, or switch with /model.", color=GRAY)
            return None

        if r.status_code != 200:
            spinner.stop()
            bullet(f"{RED}API error {r.status_code}{RESET}", RED)
            result(r.text[:1000], color=RED)
            return None

        text, shown, color = "", 0, None  # color None = undecided
        try:
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                payload = raw[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                delta = d.get("content") or ""
                if not delta:
                    if (d.get("reasoning_content") or d.get("reasoning")) and color is None:
                        spinner.label = "Reasoning"  # model is thinking before it answers
                    continue
                text += delta
                if color is None:
                    stripped = text.lstrip()
                    if len(stripped) < 6:
                        continue
                    spinner.stop()
                    color = GRAY if stripped.startswith(TOOL_PREFIXES) else ""
                    print(f"\n{ORANGE}⏺{RESET} ", end="")
                    shown = len(text) - len(stripped)
                type_out(text[shown:], color)
                shown = len(text)
        except KeyboardInterrupt:
            spinner.stop()
            print(f"\n{GRAY}  ⎿  Interrupted{RESET}")
            return None
        finally:
            spinner.stop()

        if color is None and text.strip():
            print(f"\n{ORANGE}⏺{RESET} ", end="")
            type_out(text.strip())
        print()
        if not text.strip():
            bullet(f"{RED}Empty reply{RESET} – the model returned nothing. Try again or switch /model.", RED)
        return text.strip()

    # ---------- tools ----------
    @staticmethod
    def parse_command(answer):
        m = re.search(r"^\s*WRITE_FILE:\s*(.+?)\s*\n```[^\n]*\n(.*?)\n?```", answer, re.MULTILINE | re.DOTALL)
        if m:
            return "write", (m.group(1).strip().strip('"`'), m.group(2))
        m = re.search(r"^\s*WRITE_FILE:\s*(.+?)\s*\n```[^\n]*\n(.*)$", answer, re.MULTILINE | re.DOTALL)
        if m:  # cut off before closing ```
            return "write", (m.group(1).strip().strip('"`'), m.group(2).rstrip("`\n"))
        m = re.search(r"^\s*LOAD_SKILL:\s*(.+)$", answer, re.MULTILINE)
        if m:
            return "skill", m.group(1).strip().strip('"`')
        m = re.search(r"^\s*READ_FILE:\s*(.+)$", answer, re.MULTILINE)
        if m:
            return "read", m.group(1).strip().strip('"`')
        m = re.search(r"^\s*(RUN_PS|RUN):\s*(.+)$", answer, re.MULTILINE)
        if not m:
            return None, None
        return ("ps" if m.group(1) == "RUN_PS" else "cmd"), m.group(2).strip().strip("`")

    def resolve(self, path):
        return os.path.abspath(os.path.join(self.cwd, os.path.expandvars(path)))

    def write_file(self, path, content):
        full = self.resolve(path)
        try:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content.rstrip("\n") + "\n")
            return f"Wrote {len(content.splitlines())} lines to {full}"
        except Exception as e:
            return f"[WRITE ERROR] {e}"

    def read_file(self, path):
        try:
            with open(self.resolve(path), "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            return "\n".join(f"{i + 1}: {l}" for i, l in enumerate(lines)) or "(empty file)"
        except Exception as e:
            return f"[READ ERROR] {e}"

    def execute(self, shell, command):
        m = re.match(r"^(cd|chdir|Set-Location|sl)(\s+/d)?\s+(.+)$", command, re.IGNORECASE)
        if m:
            target = os.path.expandvars(os.path.expanduser(m.group(3).strip().strip('"')))
            new = os.path.abspath(os.path.join(self.cwd, target))
            if os.path.isdir(new):
                self.cwd = new
                return f"Changed directory to {new}"
            return f"Directory not found: {new}"

        if shell == "ps" and os.name != "nt" and not shutil.which("pwsh"):
            return "[ERROR] PowerShell is not installed. Use RUN: with a Linux shell command."
        args, use_shell = (["powershell" if os.name == "nt" else "pwsh", "-NoProfile", "-Command", command], False) if shell == "ps" else (command, True)
        spinner = Spinner("Running")
        spinner.start()
        try:
            res = subprocess.run(args, shell=use_shell, cwd=self.cwd, capture_output=True,
                                 text=True, timeout=120, encoding="utf-8", errors="replace")
            out = (res.stdout or "") + (("\n" + res.stderr) if res.stderr else "")
            return f"{out.strip()}\n[EXIT CODE] {res.returncode}"
        except subprocess.TimeoutExpired:
            return "[ERROR] Command timed out after 120 seconds."
        except Exception as e:
            return f"[EXECUTION ERROR] {e}"
        finally:
            spinner.stop()

    # ---------- approval ----------
    @staticmethod
    def permission_box(title, body):
        w = width()
        print(f"\n{CYAN}╭{'─' * (w - 2)}╮{RESET}")
        print(f"{CYAN}│{RESET} {BOLD}{title}{RESET}")
        for l in body:
            print(f"{CYAN}│{RESET}   {l[:w - 6]}")
        print(f"{CYAN}╰{'─' * (w - 2)}╯{RESET}")

    def ask_permission(self, title, body):
        self.permission_box(title, body)
        c = pick("Do you want to proceed?", [
            ("Yes", ""),
            ("Yes, and don't ask again", "switches to auto-accept mode"),
            ("No, and tell it what to do differently", "esc"),
        ])
        return {0: "yes", 1: "all"}.get(c, "no")

    # ---------- main task loop ----------
    def ask(self, question):
        self.messages.append({"role": "user", "content": question})

        for step in range(self.max_steps):
            answer = self.call_api()
            if answer is None:
                if self.messages[-1]["role"] == "user":
                    self.messages.pop()
                return
            self.messages.append({"role": "assistant", "content": answer})

            kind, command = self.parse_command(answer)
            if not command:
                if answer.lstrip().startswith(TOOL_PREFIXES):
                    result("Unrecognized tool format – reply shown above", color=RED)
                elif self.mode == "plan":
                    self.offer_plan()
                return

            if kind == "skill":
                bullet(f"{BOLD}Skill{RESET}({command})")
                s = load_skills().get(command.lower())
                if s:
                    result(f"Loaded skill: {s['name']}")
                    folder = os.path.dirname(s["path"])
                    out = (f"[RESULT] Skill '{s['name']}' instructions:\n{s['body']}\n\n"
                           f"This skill's folder is {folder} - files it mentions (scripts, reference docs) "
                           f"are relative to that folder; use READ_FILE / RUN with the full path. "
                           f"Now continue the task following these instructions.")
                else:
                    result(f"No skill named '{command}'", color=RED)
                    out = f"[RESULT] No skill named '{command}'. Available: {', '.join(load_skills()) or 'none'}"
                self.messages.append({"role": "user", "content": out})
                continue

            if kind == "read":
                bullet(f"{BOLD}Read{RESET}({command})")
                output = self.read_file(command)
                result(output if "[READ ERROR]" in output else f"Read {output.count(chr(10)) + 1} lines")
                self.messages.append({"role": "user", "content": f"[RESULT]\n{output[:MAX_OUTPUT_CHARS * 3]}"})
                continue

            if self.mode == "plan":
                result("Blocked – plan mode is read-only", color=TEAL)
                self.messages.append({"role": "user", "content":
                    "[RESULT] Blocked: plan mode is read-only. Only use READ_FILE, then reply with your plan in plain text."})
                continue

            if kind == "write":
                path, content = command
                full = self.resolve(path)
                verb = "Update" if os.path.exists(full) else "Write"
                bullet(f"{BOLD}{verb}{RESET}({path})")
                title = f"{'Overwrite' if verb == 'Update' else 'Create'} {full}  ({len(content.splitlines())} lines)"
                body = []
            else:
                label = "PowerShell" if kind == "ps" else "Shell"
                bullet(f"{BOLD}{label}{RESET}({command})")
                title, body = f"{label} command", [command]

            if self.mode != "auto":
                choice = self.ask_permission(title, body)
                if choice == "all":
                    self.mode = "auto"
                    result("Auto-accept mode on", color=PURPLE)
                elif choice == "no":
                    why = input(f"{GRAY}  what should it do instead? (enter to stop) › {RESET}").strip()
                    result("User rejected", color=RED)
                    msg = "[RESULT] User declined that action."
                    if why:
                        self.messages.append({"role": "user", "content": msg + f" User says: {why}"})
                        continue
                    self.messages.append({"role": "user", "content": msg + " Stop and wait for instructions."})
                    return

            if kind == "write":
                output = self.write_file(*command)
            else:
                output = self.execute(kind, command)
            result(output, color=RED if "ERROR]" in output else GRAY)

            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n...[output truncated]"
            self.messages.append({"role": "user", "content": f"[RESULT] cwd={self.cwd}\n{output}"})

        bullet(f"{GRAY}Reached step limit ({self.max_steps}). Type 'continue' to keep going.{RESET}")

    def offer_plan(self):
        print()
        c = pick("Would you like to proceed with this plan?", [
            ("Yes, and auto-accept edits", ""),
            ("Yes, and manually approve edits", ""),
            ("No, keep planning", "tell it what to change"),
        ])
        if c in (0, 1):
            self.mode = "auto" if c == 0 else "default"
            result(f"{'Auto-accept' if c == 0 else 'Default'} mode on")
            self.ask("The plan is approved. Execute it now step by step.")


def welcome(agent):
    w = min(width(), 70)
    inner = w - 4

    def row(s, plain_len):
        print(f"{ORANGE}│{RESET} {s}{' ' * max(0, inner - plain_len)} {ORANGE}│{RESET}")

    print(f"{ORANGE}╭{'─' * (w - 2)}╮{RESET}")
    row(f"{ORANGE}✻{RESET} Welcome to {BOLD}{APP_NAME}{RESET}!", len(f"✻ Welcome to {APP_NAME}!"))
    row("", 0)
    for s in ("? for shortcuts · tab to cycle modes", f"model: {agent.model}", f"cwd: {agent.cwd}"):
        s = s[:inner - 2]
        row(f"  {GRAY}{s}{RESET}", len(s) + 2)
    print(f"{ORANGE}╰{'─' * (w - 2)}╯{RESET}")


def main():
    agent = Agent()
    box = InputBox()
    welcome(agent)

    while True:
        try:
            q = box.read(agent)
            if not q:
                continue
            cmd, _, arg = q.partition(" ")
            cmd, arg = cmd.lower(), arg.strip()

            if cmd == "/exit":
                break
            elif cmd in ("/help", "?"):
                show_shortcuts()
            elif cmd == "/clear":
                agent.reset()
                result("Conversation cleared")
            elif cmd == "/where":
                result(agent.cwd)
            elif cmd == "/cd":
                result(agent.execute("cmd", "cd " + arg) if arg else agent.cwd)
            elif cmd == "/model":
                if arg:
                    agent.model = normalize_model(arg)
                else:
                    models = fetch_local_models()
                    names = [m for m, _ in models]
                    cur = names.index(agent.model) if agent.model in names else 0
                    hint = "type an installed Ollama model name, e.g. qwen2.5:7b"
                    opts = models + [("Custom…", hint)]
                    print()
                    i = pick(f"Select model ({len(models)} free)", opts, cur)
                    if i is not None:
                        if i == len(models):
                            custom = input(f"{GRAY}  model id › {RESET}").strip()
                            if custom:
                                agent.model = normalize_model(custom)
                        else:
                            agent.model = models[i][0]
                result(f"Model: {agent.model}")
            elif cmd == "/mode":
                print()
                i = pick("Select mode", [("default", "asks before every action"),
                                         ("auto-accept", "runs and writes without asking"),
                                         ("plan", "read-only; proposes a plan first")],
                         MODES.index(agent.mode))
                if i is not None:
                    agent.mode = MODES[i]
                result(f"Mode: {agent.mode}")
            elif cmd == "/steps":
                if arg.isdigit() and int(arg) > 0:
                    agent.max_steps = int(arg)
                result(f"Max steps: {agent.max_steps}")
            elif cmd == "/save":
                agent.save(arg or None)
            elif cmd == "/load":
                agent.load(arg) if arg else result("Usage: /load <name>")
            elif cmd == "/run":
                command = input(f"{GRAY}  command › {RESET}").strip()
                if command:
                    bullet(f"{BOLD}Bash{RESET}({command})")
                    result(agent.execute("cmd", command))
            elif cmd == "/skills":
                skills = load_skills()
                bullet(f"Skills {GRAY}({SKILLS_DIR}){RESET}")
                if not skills:
                    result("No skills yet. Make one with: /skill new <name>")
                for s in skills.values():
                    print(f"     {ORANGE}/{s['name']:<20}{RESET}{GRAY}{s['description']}{RESET}")
            elif cmd == "/skill":
                sub, _, name = arg.partition(" ")
                name = re.sub(r"[^A-Za-z0-9_-]", "-", name.strip())
                if sub not in ("new", "edit") or not name:
                    result("Usage: /skill new <name>   or   /skill edit <name>")
                    continue
                os.makedirs(SKILLS_DIR, exist_ok=True)
                existing = load_skills().get(name.lower())
                path = existing["path"] if existing else os.path.join(SKILLS_DIR, name + ".md")
                if not existing:
                    if sub == "edit":
                        result(f"No skill named '{name}'. Create it with /skill new {name}", color=RED)
                        continue
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(SKILL_TEMPLATE.format(name=name))
                    result(f"Created {path}")
                if os.name == "nt":
                    subprocess.Popen(["notepad", path])
                else:
                    editor = os.environ.get("EDITOR", "nano")
                    result(f"Open this skill with: {editor} {path}")
                result("Opened in Notepad – save it and it works right away")
            elif cmd.startswith("/") and cmd[1:] in load_skills():
                s = load_skills()[cmd[1:]]
                bullet(f"{BOLD}Skill{RESET}({s['name']})")
                agent.ask(f"Use the '{s['name']}' skill. Its folder is {os.path.dirname(s['path'])} "
                          f"(helper files it mentions are relative to it). Its instructions:\n{s['body']}\n\n"
                          f"Task: {arg or 'Follow the skill instructions.'}")
            elif cmd.startswith("/"):
                result(f"Unknown command {cmd}. Type ? for shortcuts, /skills for skills.", color=RED)
            else:
                agent.ask(q)
        except KeyboardInterrupt:
            print(f"\n{GRAY}  ⎿  Interrupted · /exit to quit{RESET}")
        except EOFError:
            break


if __name__ == "__main__":
    main()