import sys
from pathlib import Path

# On Windows, stdout can fall back to the legacy cp1252 codepage when the
# process isn't attached to a real console (piped output, some IDE
# terminals), which crashes on non-ASCII text. Force UTF-8 and never crash
# on an unencodable character instead.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

LOG_LINES = []

def log(msg=""):
    print(msg)
    LOG_LINES.append(msg)

def write_report(filename="filter_design_report.txt", out_dir="filter designs"):
    Path(out_dir).mkdir(exist_ok=True)
    filepath = Path(out_dir) / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    print(f"[OK] Report saved to {filepath}")
    return filepath
