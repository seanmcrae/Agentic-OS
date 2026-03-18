"""
Knowledge Graph - Structured relational knowledge storage
==========================================================

A lightweight but powerful in-memory knowledge graph with:
- Typed nodes and edges
- Graph traversal (BFS/DFS)
- Shortest-path reasoning
- Pattern matching queries
- Inference over relationships
- Export to dict / import from dict
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class KGNode:
    """A node (entity/concept) in the knowledge graph."""
    node_id: str
    label: str                         # e.g. "Person", "Concept", "Event"
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    source: str = "unknown"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return isinstance(other, KGNode) and self.node_id == other.node_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "name": self.name,
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class KGEdge:
    """A directed edge (relationship) between two nodes."""
    edge_id: str
    source_id: str
    target_id: str
    relation: str                      # e.g. "is_a", "part_of", "causes", "related_to"
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    source: str = "unknown"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
        }


@dataclass
class GraphPath:
    """A path through the knowledge graph."""
    nodes: List[KGNode]
    edges: List[KGEdge]
    total_weight: float = 0.0

    @property
    def length(self) -> int:
        return len(self.edges)

    def to_text(self) -> str:
        """Convert path to human-readable text."""
        if not self.nodes:
            return ""
        parts = [self.nodes[0].name]
        for edge, node in zip(self.edges, self.nodes[1:]):
            parts.append(f" --[{edge.relation}]--> {node.name}")
        return "".join(parts)


@dataclass
class QueryResult:
    """Result of a knowledge graph query."""
    nodes: List[KGNode] = field(default_factory=list)
    edges: List[KGEdge] = field(default_factory=list)
    paths: List[GraphPath] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Knowledge Graph
# ============================================================================

class KnowledgeGraph:
    """
    In-memory knowledge graph with full CRUD and traversal.

    Stores entities as nodes and relationships as directed edges.
    Supports pattern queries, shortest-path reasoning, and
    inference over multi-hop relationships.

    Example:
        kg = KnowledgeGraph()
        python = kg.add_node("Language", "Python")
        guido = kg.add_node("Person", "Guido van Rossum")
        kg.add_edge(guido, python, "created")
        path = kg.shortest_path("Guido van Rossum", "Python")
    """

    def __init__(self, max_nodes: int = 1_000_000):
        self.max_nodes = max_nodes

        # Primary stores
        self._nodes: Dict[str, KGNode] = {}          # node_id -> KGNode
        self._edges: Dict[str, KGEdge] = {}          # edge_id -> KGEdge

        # Indices for fast lookup
        self._name_index: Dict[str, List[str]] = defaultdict(list)   # name -> [node_ids]
        self._label_index: Dict[str, List[str]] = defaultdict(list)  # label -> [node_ids]

        # Adjacency lists
        self._out_edges: Dict[str, List[str]] = defaultdict(list)    # node_id -> [edge_ids]
        self._in_edges: Dict[str, List[str]] = defaultdict(list)     # node_id -> [edge_ids]
        self._relation_index: Dict[str, List[str]] = defaultdict(list)  # relation -> [edge_ids]

        self._node_counter = 0
        self._edge_counter = 0

    # =========================================================================
    # Node operations
    # =========================================================================

    def add_node(
        self,
        label: str,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        source: str = "unknown",
        node_id: Optional[str] = None,
    ) -> KGNode:
        """
        Add or update a node in the graph.

        If a node with the same label+name exists, returns it (with updates).
        """
        # Check for existing node by name+label
        existing = self.find_node(name, label)
        if existing:
            # Update properties
            if properties:
                existing.properties.update(properties)
            existing.confidence = max(existing.confidence, confidence)
            existing.updated_at = time.time()
            return existing

        nid = node_id or f"n{self._node_counter:08d}"
        self._node_counter += 1

        node = KGNode(
            node_id=nid,
            label=label,
            name=name,
            properties=properties or {},
            confidence=confidence,
            source=source,
        )

        self._nodes[nid] = node
        self._name_index[name.lower()].append(nid)
        self._label_index[label].append(nid)

        logger.debug(f"KG: added node [{label}] {name}")
        return node

    def get_node(self, node_id: str) -> Optional[KGNode]:
        return self._nodes.get(node_id)

    def find_node(self, name: str, label: Optional[str] = None) -> Optional[KGNode]:
        """Find a node by name (and optional label)."""
        candidates = self._name_index.get(name.lower(), [])
        for nid in candidates:
            node = self._nodes.get(nid)
            if node and (label is None or node.label == label):
                return node
        return None

    def find_nodes(
        self,
        name: Optional[str] = None,
        label: Optional[str] = None,
        property_filter: Optional[Dict[str, Any]] = None,
        limit: int = 50,
    ) -> List[KGNode]:
        """Find multiple nodes matching criteria."""
        if label:
            candidates = [self._nodes[nid] for nid in self._label_index.get(label, []) if nid in self._nodes]
        elif name:
            candidates = [self._nodes[nid] for nid in self._name_index.get(name.lower(), []) if nid in self._nodes]
        else:
            candidates = list(self._nodes.values())

        if name and not label:
            name_lower = name.lower()
            candidates = [n for n in candidates if name_lower in n.name.lower()]

        if property_filter:
            candidates = [
                n for n in candidates
                if all(n.properties.get(k) == v for k, v in property_filter.items())
            ]

        return candidates[:limit]

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return False

        node = self._nodes.pop(node_id)

        # Remove from indices
        nid_list = self._name_index.get(node.name.lower(), [])
        if node_id in nid_list:
            nid_list.remove(node_id)

        label_list = self._label_index.get(node.label, [])
        if node_id in label_list:
            label_list.remove(node_id)

        # Remove all connected edges
        edge_ids = list(self._out_edges.get(node_id, [])) + list(self._in_edges.get(node_id, []))
        for eid in edge_ids:
            self._remove_edge_by_id(eid)

        return True

    def update_node(self, node_id: str, **kwargs) -> bool:
        """Update node properties."""
        node = self._nodes.get(node_id)
        if not node:
            return False
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
            else:
                node.properties[key] = value
        node.updated_at = time.time()
        return True

    # =========================================================================
    # Edge operations
    # =========================================================================

    def add_edge(
        self,
        source: KGNode | str,
        target: KGNode | str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        confidence: float = 1.0,
        bidirectional: bool = False,
        source_name: str = "unknown",
    ) -> KGEdge:
        """
        Add a directed edge between two nodes.

        Args:
            source: Source node or node_id
            target: Target node or node_id
            relation: Relationship type
            properties: Edge metadata
            weight: Edge weight for pathfinding
            confidence: Confidence in this relationship
            bidirectional: If True, traverse in both directions
            source_name: Data source for this edge
        """
        src_id = source.node_id if isinstance(source, KGNode) else source
        tgt_id = target.node_id if isinstance(target, KGNode) else target

        if src_id not in self._nodes or tgt_id not in self._nodes:
            raise ValueError(f"Node not found: {src_id} or {tgt_id}")

        # Check for duplicate edge
        existing = self._find_edge(src_id, tgt_id, relation)
        if existing:
            existing.confidence = max(existing.confidence, confidence)
            existing.weight = (existing.weight + weight) / 2
            if properties:
                existing.properties.update(properties)
            return existing

        eid = f"e{self._edge_counter:010d}"
        self._edge_counter += 1

        edge = KGEdge(
            edge_id=eid,
            source_id=src_id,
            target_id=tgt_id,
            relation=relation,
            properties=properties or {},
            weight=weight,
            confidence=confidence,
            bidirectional=bidirectional,
            source=source_name,
        )

        self._edges[eid] = edge
        self._out_edges[src_id].append(eid)
        self._in_edges[tgt_id].append(eid)
        self._relation_index[relation].append(eid)

        # For bidirectional edges, also index the reverse
        if bidirectional:
            self._in_edges[src_id].append(eid)
            self._out_edges[tgt_id].append(eid)

        logger.debug(f"KG: edge [{relation}] {src_id} -> {tgt_id}")
        return edge

    def get_neighbors(
        self,
        node: KGNode | str,
        relation: Optional[str] = None,
        direction: str = "out",  # "out" | "in" | "both"
    ) -> List[Tuple[KGNode, KGEdge]]:
        """Get neighboring nodes connected by edges."""
        node_id = node.node_id if isinstance(node, KGNode) else node

        edge_ids: List[str] = []
        if direction in ("out", "both"):
            edge_ids += self._out_edges.get(node_id, [])
        if direction in ("in", "both"):
            edge_ids += self._in_edges.get(node_id, [])

        results = []
        for eid in edge_ids:
            edge = self._edges.get(eid)
            if not edge:
                continue
            if relation and edge.relation != relation:
                continue

            # Determine neighbor node
            if edge.source_id == node_id:
                neighbor_id = edge.target_id
            else:
                neighbor_id = edge.source_id

            neighbor = self._nodes.get(neighbor_id)
            if neighbor:
                results.append((neighbor, edge))

        return results

    def get_edges_between(self, node_a: str, node_b: str) -> List[KGEdge]:
        """Get all edges between two nodes (any direction)."""
        result = []
        for eid in self._out_edges.get(node_a, []):
            edge = self._edges.get(eid)
            if edge and edge.target_id == node_b:
                result.append(edge)
        for eid in self._out_edges.get(node_b, []):
            edge = self._edges.get(eid)
            if edge and edge.target_id == node_a:
                result.append(edge)
        return result

    # =========================================================================
    # Traversal & Pathfinding
    # =========================================================================

    def shortest_path(
        self,
        source_name: str,
        target_name: str,
        relation_filter: Optional[List[str]] = None,
        max_hops: int = 6,
    ) -> Optional[GraphPath]:
        """
        Find the shortest path between two nodes using BFS.

        Args:
            source_name: Name of the starting node
            target_name: Name of the destination node
            relation_filter: Only traverse edges with these relations
            max_hops: Maximum path length

        Returns:
            GraphPath if found, None otherwise
        """
        source = self.find_node(source_name)
        target = self.find_node(target_name)

        if not source or not target:
            logger.debug(f"Pathfinding: node not found ({source_name} or {target_name})")
            return None

        if source.node_id == target.node_id:
            return GraphPath(nodes=[source], edges=[])

        # BFS
        queue: deque = deque([(source, [source], [])])
        visited: Set[str] = {source.node_id}

        while queue:
            current, path_nodes, path_edges = queue.popleft()

            if len(path_edges) >= max_hops:
                continue

            for neighbor, edge in self.get_neighbors(current, direction="both"):
                if relation_filter and edge.relation not in relation_filter:
                    continue

                if neighbor.node_id == target.node_id:
                    return GraphPath(
                        nodes=path_nodes + [neighbor],
                        edges=path_edges + [edge],
                        total_weight=sum(e.weight for e in path_edges + [edge]),
                    )

                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    queue.append((
                        neighbor,
                        path_nodes + [neighbor],
                        path_edges + [edge],
                    ))

        return None

    def all_paths(
        self,
        source_name: str,
        target_name: str,
        max_hops: int = 4,
        max_results: int = 5,
    ) -> List[GraphPath]:
        """Find all paths between two nodes (up to max_hops)."""
        source = self.find_node(source_name)
        target = self.find_node(target_name)

        if not source or not target:
            return []

        results = []

        def dfs(current: KGNode, target_id: str, path_nodes: List, path_edges: List, visited: Set):
            if current.node_id == target_id:
                results.append(GraphPath(
                    nodes=list(path_nodes),
                    edges=list(path_edges),
                    total_weight=sum(e.weight for e in path_edges),
                ))
                return

            if len(path_edges) >= max_hops or len(results) >= max_results:
                return

            for neighbor, edge in self.get_neighbors(current, direction="out"):
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    path_nodes.append(neighbor)
                    path_edges.append(edge)
                    dfs(neighbor, target_id, path_nodes, path_edges, visited)
                    path_nodes.pop()
                    path_edges.pop()
                    visited.discard(neighbor.node_id)

        dfs(source, target.node_id, [source], [], {source.node_id})
        return sorted(results, key=lambda p: p.total_weight)

    def bfs(
        self,
        start_name: str,
        max_depth: int = 3,
        relation_filter: Optional[List[str]] = None,
    ) -> List[KGNode]:
        """BFS traversal from a starting node."""
        start = self.find_node(start_name)
        if not start:
            return []

        visited = {start.node_id}
        queue: deque = deque([(start, 0)])
        result = [start]

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for neighbor, edge in self.get_neighbors(current, direction="out"):
                if relation_filter and edge.relation not in relation_filter:
                    continue
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))

        return result

    # =========================================================================
    # Pattern Matching & Queries
    # =========================================================================

    def query(
        self,
        pattern: str,
        limit: int = 20,
    ) -> QueryResult:
        """
        Natural language-style pattern query.

        Patterns:
          "?X is_a Animal"       -> find all things that are animals
          "Python related_to ?"  -> find what Python is related to
          "? created ?"          -> find all creator relationships
        """
        parts = pattern.strip().split()
        if len(parts) != 3:
            return QueryResult()

        subj, pred, obj = parts
        subj_var = subj == "?"
        obj_var = obj == "?"
        pred_var = pred == "?"

        results = QueryResult()

        if pred_var:
            # Find all relations between subj and obj
            s_node = self.find_node(subj) if not subj_var else None
            o_node = self.find_node(obj) if not obj_var else None
            if s_node and o_node:
                edges = self.get_edges_between(s_node.node_id, o_node.node_id)
                results.edges = edges[:limit]
            return results

        # Filter edges by relation
        relation_edges = [
            self._edges[eid]
            for eid in self._relation_index.get(pred, [])
            if eid in self._edges
        ]

        for edge in relation_edges[:limit * 2]:
            src = self._nodes.get(edge.source_id)
            tgt = self._nodes.get(edge.target_id)
            if not src or not tgt:
                continue

            match = True
            if not subj_var and src.name.lower() != subj.lower():
                match = False
            if not obj_var and tgt.name.lower() != obj.lower():
                match = False

            if match:
                if subj_var and src not in results.nodes:
                    results.nodes.append(src)
                if obj_var and tgt not in results.nodes:
                    results.nodes.append(tgt)
                results.edges.append(edge)

                if len(results.nodes) >= limit:
                    break

        return results

    def find_common_neighbors(
        self,
        node_a_name: str,
        node_b_name: str,
    ) -> List[KGNode]:
        """Find nodes connected to both node_a and node_b."""
        a = self.find_node(node_a_name)
        b = self.find_node(node_b_name)
        if not a or not b:
            return []

        a_neighbors = {n.node_id for n, _ in self.get_neighbors(a, direction="both")}
        b_neighbors = {n.node_id for n, _ in self.get_neighbors(b, direction="both")}

        common_ids = a_neighbors & b_neighbors
        return [self._nodes[nid] for nid in common_ids if nid in self._nodes]

    # =========================================================================
    # Inference
    # =========================================================================

    def infer_transitive(
        self,
        relation: str,
        source_name: str,
        max_depth: int = 5,
    ) -> List[Tuple[KGNode, int]]:
        """
        Infer all nodes reachable by transitive closure of a relation.

        Example: if A is_a B and B is_a C, then A is_a C (transitively).

        Returns list of (node, depth) tuples.
        """
        source = self.find_node(source_name)
        if not source:
            return []

        visited: Dict[str, int] = {source.node_id: 0}
        queue: deque = deque([(source, 0)])
        results = []

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for neighbor, edge in self.get_neighbors(current, relation=relation, direction="out"):
                if neighbor.node_id not in visited:
                    visited[neighbor.node_id] = depth + 1
                    results.append((neighbor, depth + 1))
                    queue.append((neighbor, depth + 1))

        return results

    def infer_inverse(
        self,
        relation: str,
        inverse_relation: str,
    ) -> List[KGEdge]:
        """
        Infer inverse edges: if A --[relation]--> B, then B --[inverse]--> A.
        Returns the list of inferred edges (not yet added to graph).
        """
        inferred = []
        for eid in self._relation_index.get(relation, []):
            edge = self._edges.get(eid)
            if edge:
                # Check if inverse already exists
                if not self._find_edge(edge.target_id, edge.source_id, inverse_relation):
                    inferred.append(KGEdge(
                        edge_id=f"inferred_{eid}",
                        source_id=edge.target_id,
                        target_id=edge.source_id,
                        relation=inverse_relation,
                        confidence=edge.confidence * 0.9,  # Slightly less confident
                        source="inferred",
                    ))
        return inferred

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def bulk_add(self, triples: List[Tuple[str, str, str, str, str]]) -> int:
        """
        Add multiple triples at once.

        Args:
            triples: List of (source_label, source_name, relation, target_label, target_name)

        Returns:
            Number of triples added
        """
        count = 0
        for src_label, src_name, relation, tgt_label, tgt_name in triples:
            try:
                src = self.add_node(src_label, src_name)
                tgt = self.add_node(tgt_label, tgt_name)
                self.add_edge(src, tgt, relation)
                count += 1
            except Exception as e:
                logger.warning(f"Skipping triple ({src_name}, {relation}, {tgt_name}): {e}")
        return count

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Export the entire graph to a dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "metadata": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "exported_at": time.time(),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Import a graph from a dictionary."""
        kg = cls()
        for node_data in data.get("nodes", []):
            node = KGNode(**{k: v for k, v in node_data.items() if k != "embedding"})
            kg._nodes[node.node_id] = node
            kg._name_index[node.name.lower()].append(node.node_id)
            kg._label_index[node.label].append(node.node_id)

        for edge_data in data.get("edges", []):
            edge = KGEdge(**edge_data)
            kg._edges[edge.edge_id] = edge
            kg._out_edges[edge.source_id].append(edge.edge_id)
            kg._in_edges[edge.target_id].append(edge.edge_id)
            kg._relation_index[edge.relation].append(edge.edge_id)

        return kg

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        relation_counts = {rel: len(eids) for rel, eids in self._relation_index.items()}
        label_counts = {label: len(nids) for label, nids in self._label_index.items()}

        # Degree statistics
        degrees = [len(self._out_edges.get(nid, [])) for nid in self._nodes]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0

        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "labels": label_counts,
            "relations": relation_counts,
            "avg_out_degree": avg_degree,
            "max_degree": max(degrees) if degrees else 0,
        }

    def _find_edge(self, src_id: str, tgt_id: str, relation: str) -> Optional[KGEdge]:
        """Check if a specific edge already exists."""
        for eid in self._out_edges.get(src_id, []):
            edge = self._edges.get(eid)
            if edge and edge.target_id == tgt_id and edge.relation == relation:
                return edge
        return None

    def _remove_edge_by_id(self, edge_id: str) -> None:
        """Remove an edge by ID, cleaning up all indices."""
        edge = self._edges.pop(edge_id, None)
        if not edge:
            return

        for eid_list in self._out_edges.values():
            if edge_id in eid_list:
                eid_list.remove(edge_id)

        for eid_list in self._in_edges.values():
            if edge_id in eid_list:
                eid_list.remove(edge_id)

        rel_list = self._relation_index.get(edge.relation, [])
        if edge_id in rel_list:
            rel_list.remove(edge_id)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, name: str) -> bool:
        return bool(self.find_node(name))
