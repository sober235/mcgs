#!/usr/bin/env python3
"""refresh_readme_tables.py <topic_dir> -- re-inject logs/readme_tables.md into README.md between the TABLES markers."""
import sys
from pathlib import Path
T = Path(sys.argv[1]); r = T / "README.md"; s = r.read_text(); t = (T / "logs/readme_tables.md").read_text().strip()
a, b = s.index("<!-- TABLES:BEGIN -->") + len("<!-- TABLES:BEGIN -->"), s.index("<!-- TABLES:END -->")
r.write_text(s[:a] + "\n" + t + "\n" + s[b:]); print("refreshed", r)
