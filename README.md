# stale-magic

Makefile-like **Jupyter / IPython cell magic** that conditionally runs a cell based on input/output timestamps **and** keeps notebook output visually unchanged when a cell is skipped.

## Install

```bash
git clone https://github.com/scovit/stale-magic.git
cd stale-magic
pip install -e .
```

## Load the extension

In a notebook (or IPython):

```python
%load_ext stale_magic
```

## Use

Syntax:

```python
%%rule OUT1 [OUT2 ...] : [IN1 IN2 ...]
# your Python code that (re)builds OUT* from IN*
```

Inputs can include globs, including `**`:

```python
%%rule build/features.parquet : data/raw/*.csv src/**/*.py
import pandas as pd
# ... expensive build ...
```

### What “up-to-date” means

A cell is **skipped** when:

1. All outputs exist, and
2. No input is newer than the **oldest** output, and
3. The cell body hasn’t changed (SHA-256 of normalized cell text).

Otherwise, the cell runs.

## Cache

Per-rule cache files are stored under:

```
.stale_cache/<rule_id>.json
```

The cache stores:

- Cell hash
- Last execution count (best-effort)

You can delete `.stale_cache/` any time to reset.

## Metadata

`stale-magic` also stores best-effort metadata into the current cell under the key:

- `cell.metadata["stale_magic"]`

This can be useful for debugging in the notebook UI.

## License

CC-BY-SA.
