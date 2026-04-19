from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graph.graph_store import GraphStore

store = GraphStore(
    uri="neo4j://127.0.0.1:7687",
    user="neo4j",
    password="neo4j1234"
)

store.init_schema()
print("Connected successfully!")