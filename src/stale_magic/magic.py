from __future__ import annotations

import glob
import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.display import Javascript, display


# ----------------------------
# Helpers
# ----------------------------

def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def _norm_cell_text(cell: str) -> str:
    # Normalize line endings so Windows vs Unix doesn't cause spurious rebuilds
    return cell.replace("\r\n", "\n").replace("\r", "\n")


def _cell_hash(cell: str) -> str:
    return hashlib.sha256(_norm_cell_text(cell).encode("utf-8")).hexdigest()


def _parse_rule(line: str) -> Tuple[List[str], List[str]]:
    """
    Parse syntax:
        OUT1 [OUT2 ...] : [IN1 IN2 ...]

    Supports quoting via shlex:
        "out file.parquet" : "in file.csv"

    Returns:
      - outputs as raw strings (so we can expand globs later)
      - inputs as raw strings (so we can expand globs later)
    """
    tokens = shlex.split(line)
    if ":" not in tokens:
        raise ValueError(
            "Usage: %%rule OUT1 [OUT2 ...] : [IN1 IN2 ...]\n"
            'Example: %%rule "build/out 1.parquet" build/out2.json : data/raw/*.csv data/**/*.json'
        )
    i = tokens.index(":")
    out_tokens = tokens[:i]
    in_tokens = tokens[i + 1 :]

    outputs_raw = list(out_tokens)
    inputs_raw = list(in_tokens)
    return outputs_raw, inputs_raw


def _expand_globs(globs_raw: List[str]) -> List[Path]:
    """
    Expand glob patterns in inputs. Supports ** (recursive) via glob.glob(..., recursive=True).
    - If a token matches nothing and contains glob characters, raises FileNotFoundError.
    - If a token has no glob characters, keeps it as-is.
    """
    expanded: List[Path] = []

    def has_glob(s: str) -> bool:
        return any(ch in s for ch in ["*", "?", "["]) or "**" in s

    for tok in globs_raw:
        if has_glob(tok):
            matches = glob.glob(tok, recursive=True)
            if not matches:
                raise FileNotFoundError(f"Input glob matched nothing: {tok}")
            expanded.extend(Path(m) for m in matches)
        else:
            expanded.append(Path(tok))

    # Canonicalize + de-dup + stable order
    # absolute() avoids resolve() failures on weird paths; also keeps comparison stable.
    uniq = {}
    for p in expanded:
        ap = p.expanduser().absolute()
        uniq[str(ap)] = ap
    return [uniq[k] for k in sorted(uniq.keys())]


def _rule_id(outputs: List[Path], inputs: List[Path]) -> str:
    # Stable ID based on *expanded* declared deps (absolute paths, sorted).
    key = {
        "outputs": sorted(str(p.expanduser().absolute()) for p in outputs),
        "inputs": sorted(str(p.expanduser().absolute()) for p in inputs),
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _cache_dir() -> Path:
    return Path(".stale_cache")


def _cache_path(rule_id: str) -> Path:
    return _cache_dir() / f"{rule_id}.json"


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _set_cell_metadata(key: str, value) -> None:
    """
    Best-effort: ask the frontend to store data into current cell metadata.
    Works in classic Notebook and many Notebook 7 / Lab contexts.
    """
    value_js = json.dumps(value)
    key_js = json.dumps(key)

    js = f"""
    (function() {{
      try {{
        // Classic Notebook
        if (window.Jupyter && Jupyter.notebook && Jupyter.notebook.get_selected_cell) {{
          var cell = Jupyter.notebook.get_selected_cell();
          if (cell && cell.metadata) {{
            cell.metadata[{key_js}] = {value_js};
            return;
          }}
        }}

        // Notebook 7 / JupyterLab-ish best-effort
        var app = window.jupyterapp || window.jupyterApp || null;
        if (app && app.shell && app.shell.currentWidget && app.shell.currentWidget.content) {{
          var activeCell = app.shell.currentWidget.content.activeCell;
          if (activeCell && activeCell.model && activeCell.model.metadata && activeCell.model.metadata.set) {{
            activeCell.model.metadata.set({key_js}, {value_js});
            return;
          }}
          if (activeCell && activeCell.model) {{
            activeCell.model.metadata = activeCell.model.metadata || {{}};
            activeCell.model.metadata[{key_js}] = {value_js};
            return;
          }}
        }}

        console.warn("stale_magic: Could not set cell metadata (unsupported frontend context).");
      }} catch (e) {{
        console.warn("stale_magic: Error setting cell metadata:", e);
      }}
    }})();
    """
    display(Javascript(js))


@dataclass
class Decision:
    run: bool
    reason: str


def _decide(outputs: List[Path], inputs: List[Path], cell_hash: str, cache: dict | None) -> Decision:
    missing_outs = [p for p in outputs if not p.exists()]
    if missing_outs:
        return Decision(True, f"missing output(s): {', '.join(map(str, missing_outs))}")

    out_mtimes = [(p, _mtime(p)) for p in outputs]
    oldest_out, oldest_mtime = min(out_mtimes, key=lambda t: t[1])

    newer_inputs = [p for p in inputs if _mtime(p) > oldest_mtime]
    if newer_inputs:
        newest_in = max(newer_inputs, key=_mtime)
        return Decision(True, f"input newer than oldest output: {newest_in} > {oldest_out}")

    old_hash = (cache or {}).get("cell_hash")
    if old_hash != cell_hash:
        if old_hash is None:
            return Decision(True, "no cached cell hash (first run for this rule)")
        return Decision(True, "cell content changed (hash differs)")

    return Decision(False, "all outputs up-to-date and cell unchanged")


# ----------------------------
# Magics
# ----------------------------

@magics_class
class StaleMagics(Magics):
    """
    Usage:
        %%rule OUT1 [OUT2 ...] : IN1 [IN2 ...]   (inputs can include globs like data/*.csv or data/**/x.json)

    Stores robust state in:
        .stale_cache/<rule_id>.json

    Also writes best-effort info into cell metadata under:
        "stale_magic": {...}
    """

    @cell_magic
    def rule(self, line: str, cell: str):
        outputs_raw, inputs_raw = _parse_rule(line)
        outputs = _expand_globs(outputs_raw)
        inputs = _expand_globs(inputs_raw)

        # Validate inputs exist (post-expansion)
        missing_inputs = [str(p) for p in inputs if not p.exists()]
        if missing_inputs:
            raise FileNotFoundError(f"Missing input file(s): {', '.join(missing_inputs)}")

        rid = _rule_id(outputs, inputs)
        cache_path = _cache_path(rid)
        cache = _read_cache(cache_path)

        chash = _cell_hash(cell)
        decision = _decide(outputs, inputs, chash, cache)

        meta_payload = {
            "rule_id": rid,
            "outputs": [str(p) for p in outputs],
            "inputs": [str(p) for p in inputs],         # expanded list
            "inputs_raw": list(inputs_raw),             # original tokens (incl globs)
            "cell_hash": chash,
            "will_run": decision.run,
            "reason": decision.reason,
        }
        _set_cell_metadata("stale_magic", meta_payload)

        # --- SKIP ---
        if not decision.run:
            return None

        # --- RUN ---
        result = self.shell.run_cell(cell)

        if getattr(result, "success", True):
            payload = {
                "rule_id": rid,
                "outputs": sorted(str(p.expanduser().absolute()) for p in outputs),
                "inputs": sorted(str(p.expanduser().absolute()) for p in inputs),
                "inputs_raw": list(inputs_raw),
                "cell_hash": chash
            }
            _write_cache(cache_path, payload)

            meta_payload2 = dict(meta_payload)
            meta_payload2["ran"] = True
            _set_cell_metadata("stale_magic", meta_payload2)
        else:
            None

        return None


def load_ipython_extension(ipython):
    """IPython extension entrypoint: %load_ext stale_magic"""
    ipython.register_magics(StaleMagics)