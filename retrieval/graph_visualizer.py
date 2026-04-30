"""
Graph traversal visualizer for the GraphRAG pipeline.

Generates two representations of the retrieved subgraph:
  1. ANSI color-coded console tree — immediate visual inspection during development.
  2. Mermaid diagram string — can be pasted into any Mermaid renderer.

Both outputs use the GraphTraversalResult populated by graph_retriever.py.

Color coding (ANSI)
-------------------
  Green   — high-confidence edges (confidence ≥ 0.8)
  Yellow  — medium-confidence edges (0.6 – 0.8)
  Red     — low-confidence edges (< 0.6)
  Cyan    — entity node labels
  Purple  — relation types
  White   — path text

Usage
-----
    visualizer = GraphVisualizer()
    visualizer.print_graph(traversal_result, query_entities=["Gradient Descent"])
    mermaid_str = visualizer.to_mermaid(traversal_result)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .retrieval_context import GraphTraversalResult
from .retrieval_logger import _C   # reuse ANSI codes


class GraphVisualizer:
    """
    Console and Mermaid visualizer for graph traversal results.

    Args
    ----
    max_nodes   : Maximum nodes to display (default 20 — beyond this the
                  console output becomes unreadable).
    max_rels    : Maximum relationships to display (default 30).
    verbose     : Print visualizer diagnostics.
    """

    def __init__(
        self,
        max_nodes : int  = 20,
        max_rels  : int  = 30,
        verbose   : bool = True,
    ):
        self.max_nodes = max_nodes
        self.max_rels  = max_rels
        self.verbose   = verbose

    # Console output

    def print_graph(
        self,
        traversal      : GraphTraversalResult,
        query_entities : Optional[List[str]] = None,
    ) -> None:
        """
        Print a color-coded subgraph to the console.

        Layout
        ------
        Nodes are listed first with their type and description.
        Relationships are shown as:
            SOURCE_NAME --[RELATION_TYPE]--> TARGET_NAME   (conf: 0.85)
        Query-matching entities are highlighted in bold.
        """
        if not traversal.nodes and not traversal.relationships:
            print(f"  {_C.YELLOW}No graph results to visualise{_C.RESET}")
            return

        query_names = {n.lower() for n in (query_entities or [])}

        bar = "─" * 60
        print(f"\n{_C.CYAN}{bar}{_C.RESET}")
        print(f"{_C.CYAN}{_C.BOLD}  GRAPH TRAVERSAL VISUALISATION{_C.RESET}")
        print(f"{_C.CYAN}  depth={traversal.traversal_depth}  "
              f"nodes={len(traversal.nodes)}  "
              f"rels={len(traversal.relationships)}")
        if traversal.cypher_used:
            print(f"  Cypher: {_C.DIM}{traversal.cypher_used[:80]}…{_C.RESET}")
        print(f"{_C.CYAN}{bar}{_C.RESET}")

        # Nodes
        nodes_to_show = traversal.nodes[:self.max_nodes]
        if traversal.nodes:
            print(f"\n  {_C.WHITE}Nodes ({len(traversal.nodes)}"
                  + (f", showing {len(nodes_to_show)}" if len(traversal.nodes) > self.max_nodes else "")
                  + f"):{_C.RESET}")

        node_id_to_name: Dict[str, str] = {}
        for node in nodes_to_show:
            name  = node.get("name", "Unknown")
            ntype = node.get("entity_type", "Entity")
            desc  = node.get("description", "")[:60]
            nid   = node.get("node_id", "")
            node_id_to_name[nid] = name

            # Highlight query-matched nodes
            is_matched = name.lower() in query_names
            name_str   = (
                f"{_C.BOLD}{_C.GREEN}{name}{_C.RESET}"
                if is_matched else
                f"{_C.CYAN}{name}{_C.RESET}"
            )
            match_tag = f" {_C.GREEN}★{_C.RESET}" if is_matched else ""

            print(
                f"    {name_str}{match_tag}"
                f"  {_C.DIM}[{ntype}]{_C.RESET}"
            )
            if desc:
                print(f"      {_C.DIM}{desc}…{_C.RESET}")

        if len(traversal.nodes) > self.max_nodes:
            print(
                f"    {_C.DIM}… {len(traversal.nodes) - self.max_nodes} "
                f"more nodes not shown{_C.RESET}"
            )

        # Relationships
        rels_to_show = traversal.relationships[:self.max_rels]
        if traversal.relationships:
            print(f"\n  {_C.WHITE}Relationships ({len(traversal.relationships)}"
                  + (f", showing {len(rels_to_show)}" if len(traversal.relationships) > self.max_rels else "")
                  + f"):{_C.RESET}")

        for rel in rels_to_show:
            src_id   = rel.get("source_id", "")
            tgt_id   = rel.get("target_id", "")
            rel_type = rel.get("relation_type", rel.get("type", "RELATED"))
            conf     = float(rel.get("confidence", 1.0))
            desc     = rel.get("description", "")[:50]

            src_name = node_id_to_name.get(src_id, rel.get("source_name", src_id[:8]))
            tgt_name = node_id_to_name.get(tgt_id, rel.get("target_name", tgt_id[:8]))

            # Color by confidence
            if conf >= 0.8:
                edge_color = _C.GREEN
                conf_str   = f"{_C.GREEN}{conf:.2f}{_C.RESET}"
            elif conf >= 0.6:
                edge_color = _C.YELLOW
                conf_str   = f"{_C.YELLOW}{conf:.2f}{_C.RESET}"
            else:
                edge_color = _C.RED
                conf_str   = f"{_C.RED}{conf:.2f}{_C.RESET}"

            print(
                f"    {_C.CYAN}{src_name}{_C.RESET} "
                f"{edge_color}--[{rel_type}]-->{_C.RESET} "
                f"{_C.CYAN}{tgt_name}{_C.RESET}"
                f"  conf={conf_str}"
            )
            if desc:
                print(f"      {_C.DIM}{desc}…{_C.RESET}")

        if len(traversal.relationships) > self.max_rels:
            print(
                f"    {_C.DIM}… {len(traversal.relationships) - self.max_rels} "
                f"more relationships not shown{_C.RESET}"
            )

        # Paths 
        if traversal.paths:
            print(f"\n  {_C.WHITE}Paths found ({len(traversal.paths)}):{_C.RESET}")
            for i, path in enumerate(traversal.paths[:5], 1):
                path_str = self._path_to_str(path, node_id_to_name)
                print(f"    [{i}] {_C.PURPLE}{path_str}{_C.RESET}")

        print(f"{_C.CYAN}{bar}{_C.RESET}\n")

    # Mermaid output

    def to_mermaid(
        self,
        traversal      : GraphTraversalResult,
        title          : str = "Graph Traversal",
        query_entities : Optional[List[str]] = None,
    ) -> str:
        """
        Generate a Mermaid flowchart string for the traversal subgraph.

        The output can be pasted into:
          - https://mermaid.live
          - Any Markdown renderer that supports ```mermaid blocks
          - The Claude.ai artifacts viewer

        Returns
        -------
        Mermaid diagram string.
        """
        query_names = {n.lower() for n in (query_entities or [])}
        lines       = ["graph LR"]

        # Build node id → safe Mermaid id mapping
        node_id_map: Dict[str, str] = {}
        for i, node in enumerate(traversal.nodes[:self.max_nodes]):
            nid            = node.get("node_id", f"n{i}")
            safe_id        = f"N{i}"
            node_id_map[nid] = safe_id

            name   = _mermaid_escape(node.get("name", f"Node{i}"))
            ntype  = node.get("entity_type", "Entity")
            label  = f"{name}\n[{ntype}]"

            if node.get("name", "").lower() in query_names:
                # Highlighted node
                lines.append(f'    {safe_id}(["{label}"]):::queryNode')
            else:
                lines.append(f'    {safe_id}["{label}"]')

        # Relationships
        for rel in traversal.relationships[:self.max_rels]:
            src_id = node_id_map.get(rel.get("source_id", ""), "")
            tgt_id = node_id_map.get(rel.get("target_id", ""), "")
            if not (src_id and tgt_id):
                continue

            rel_type = _mermaid_escape(
                rel.get("relation_type", rel.get("type", "RELATED"))
            )
            conf     = float(rel.get("confidence", 1.0))

            # Line style by confidence
            if conf >= 0.8:
                arrow = f'--"{rel_type}"-->'
            elif conf >= 0.6:
                arrow = f'-."{rel_type}".->'
            else:
                arrow = f'=="{rel_type}"==>'

            lines.append(f"    {src_id} {arrow} {tgt_id}")

        # Style definitions
        lines.append("    classDef queryNode fill:#d4edda,stroke:#28a745,color:#155724")

        return "\n".join(lines)

    def print_mermaid(
        self,
        traversal      : GraphTraversalResult,
        query_entities : Optional[List[str]] = None,
    ) -> None:
        """Print the Mermaid diagram wrapped in a code fence."""
        mermaid = self.to_mermaid(traversal, query_entities=query_entities)
        print(f"\n```mermaid\n{mermaid}\n```\n")

    # Helpers 

    @staticmethod
    def _path_to_str(path: List[Dict], node_id_map: Dict[str, str]) -> str:
        parts = []
        for item in path:
            if item.get("type") == "node":
                name = item.get("name", node_id_map.get(item.get("node_id", ""), "?"))
                parts.append(name)
            elif item.get("type") == "relationship":
                parts.append(f"→[{item.get('relation_type', '?')}]→")
        return " ".join(parts) if parts else "(empty path)"


def _mermaid_escape(text: str) -> str:
    """Escape characters that break Mermaid label syntax."""
    text = text.replace('"', "'").replace("[", "(").replace("]", ")")
    text = re.sub(r"[<>{}|]", "_", text)
    return text[:40]  # Mermaid label length limit