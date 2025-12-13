"""
Adaptive tree pruning for module location IDs.

When a model has more module paths than available location IDs (1024),
this tree semantically collapses subtrees instead of using random hash
collisions. This preserves the most useful granularity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .registry import ErrorLocation


class LocationTree:
    """
    Adaptive tree pruning for module location IDs.
    
    When a model has more module paths than available location IDs (1024),
    this tree semantically collapses subtrees instead of using random hash
    collisions. This preserves the most useful granularity.
    
    Algorithm:
        1. Build tree from module paths (e.g., "encoder.layer0.attn.q")
        2. If total leaves > max_locations, find prune depth
        3. Collapse subtrees at prune depth to single IDs
        4. Query uses longest-prefix matching
    
    Example:
        tree = LocationTree(max_locations=1024)
        
        # Add paths from FX tracing
        for path in ["encoder.layer0.attn.q", "encoder.layer0.attn.k", ...]:
            tree.add_path(path)
        
        # Build IDs with auto-pruning
        tree.build_ids()
        
        # Query (may return parent ID if pruned)
        loc_id = tree.get_location("encoder.layer0.attn.q")
        
        # Print tree structure
        print(tree)
    
    Attributes:
        max_locations (int): Maximum number of location IDs
        root (Dict[str, Any]): Nested dict representing tree
        path_to_id (Dict[str, int]): Mapping from path to location ID
    """
    
    def __init__(self, max_locations: int = 1024) -> None:
        """
        Initialize the tree.
        
        Args:
            max_locations (int): Maximum number of location IDs (default 1024 for 10-bit)
        """
        self.max_locations: int = max_locations
        self.root: Dict[str, Any] = {}
        self.path_to_id: Dict[str, int] = {}
        self.__built: bool = False
        self.__start_id: int = 1
    
    def add_path(self, path: str) -> None:
        """
        Add a module path to the tree.
        
        Args:
            path (str): Dotted module path (e.g., "encoder.layers.0.conv")
        """
        if self.__built:
            raise RuntimeError("Cannot add paths after build_ids() is called")
        
        parts = path.split(".")
        node = self.root
        for part in parts:
            if part not in node:
                node[part] = {}
            node = node[part]
    
    def count_leaves(self, node: Optional[Dict[str, Any]] = None) -> int:
        """
        Count leaf nodes in subtree.
        
        Args:
            node (Optional[Dict[str, Any]]): Subtree root (None = entire tree)
            
        Returns:
            (int): Number of leaf nodes
        """
        if node is None:
            node = self.root
        if not node:
            return 1
        return sum(self.count_leaves(child) for child in node.values())
    
    def build_ids(self, start_id: int = 1) -> None:
        """
        Assign IDs, pruning if necessary to fit in max_locations.
        
        Args:
            start_id (int): First ID to use (default 1, after UNKNOWN)
        """
        if self.__built:
            return
        
        self.__start_id = start_id
        total_leaves = self.count_leaves()
        
        if total_leaves <= self.max_locations:
            self.__assign_leaf_ids(self.root, "")
        else:
            depth = self.__find_prune_depth()
            self.__assign_pruned_ids(self.root, "", depth)
        
        self.__built = True
    
    def __find_prune_depth(self) -> int:
        """
        Binary search for max depth that fits in budget.
        
        Returns:
            (int): Maximum depth to expand to before collapsing
        """
        for depth in range(20, 0, -1):
            count = self.__count_at_depth(self.root, 0, depth)
            if count <= self.max_locations:
                return depth
        return 1
    
    def __count_at_depth(self, node: Dict[str, Any], current: int, max_depth: int) -> int:
        """
        Count nodes at or below max_depth.
        
        Args:
            node (Dict[str, Any]): Current subtree
            current (int): Current depth
            max_depth (int): Maximum depth to count
            
        Returns:
            (int): Number of nodes that would get IDs at this depth
        """
        if current >= max_depth or not node:
            return 1
        return sum(
            self.__count_at_depth(child, current + 1, max_depth)
            for child in node.values()
        )
    
    def __assign_leaf_ids(self, node: Dict[str, Any], path: str) -> None:
        """
        Assign unique ID to each leaf (no pruning needed).
        
        Args:
            node (Dict[str, Any]): Current subtree
            path (str): Path to current node
        """
        if not node:
            loc_id = len(self.path_to_id) + self.__start_id
            self.path_to_id[path] = loc_id
        else:
            for name, child in node.items():
                child_path = f"{path}.{name}" if path else name
                self.__assign_leaf_ids(child, child_path)
    
    def __assign_pruned_ids(self, node: Dict[str, Any], path: str, max_depth: int, depth: int = 0) -> None:
        """
        Assign IDs, collapsing anything beyond max_depth.
        
        Args:
            node (Dict[str, Any]): Current subtree
            path (str): Path to current node
            max_depth (int): Maximum depth before collapsing
            depth (int): Current depth
        """
        if depth >= max_depth or not node:
            loc_id = len(self.path_to_id) + self.__start_id
            self.path_to_id[path] = loc_id
            self.__map_descendants(node, path, loc_id)
        else:
            for name, child in node.items():
                child_path = f"{path}.{name}" if path else name
                self.__assign_pruned_ids(child, child_path, max_depth, depth + 1)
    
    def __map_descendants(self, node: Dict[str, Any], prefix: str, loc_id: int) -> None:
        """
        Recursively map all descendant paths to the same location ID.
        
        Args:
            node (Dict[str, Any]): Subtree root
            prefix (str): Path prefix
            loc_id (int): Location ID to assign to all descendants
        """
        if not node:
            return
        for name, child in node.items():
            child_path = f"{prefix}.{name}" if prefix else name
            self.path_to_id[child_path] = loc_id
            self.__map_descendants(child, child_path, loc_id)
    
    def get_location(self, path: str) -> int:
        """
        Get location ID for a path (may be collapsed to parent).
        
        Uses longest-prefix matching to handle paths not explicitly
        in the tree (e.g., querying "encoder.layer0.attn.q" when
        only "encoder.layer0" was assigned an ID).
        
        Args:
            path (str): Module path to look up
            
        Returns:
            (int): Location ID (may be parent's ID if pruned)
        """
        if not self.__built:
            raise RuntimeError("Must call build_ids() before get_location()")
        
        if path in self.path_to_id:
            return self.path_to_id[path]
        
        while path:
            if path in self.path_to_id:
                return self.path_to_id[path]
            parts = path.split(".")
            if len(parts) == 1:
                break
            path = ".".join(parts[:-1])
        
        return ErrorLocation.UNKNOWN
    
    def register_all(self) -> Dict[str, int]:
        """
        Register all paths in this tree with the global ErrorLocation registry.
        
        Returns:
            (Dict[str, int]): Dict mapping path -> location ID (from registry)
        """
        result: Dict[str, int] = {}
        for path in self.path_to_id:
            result[path] = ErrorLocation.register(path)
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # TREE VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def __repr__(self) -> str:
        """Short string representation."""
        status = "built" if self.__built else "not built"
        unique_ids = len(set(self.path_to_id.values())) if self.__built else 0
        return f"LocationTree(leaves={self.count_leaves()}, max={self.max_locations}, ids={unique_ids}, {status})"
    
    def __str__(self) -> str:
        """Detailed tree structure visualization."""
        lines: List[str] = []
        lines.append(f"LocationTree (max_locations={self.max_locations})")
        lines.append(f"  Leaves: {self.count_leaves()}")
        if self.__built:
            unique_ids = len(set(self.path_to_id.values()))
            lines.append(f"  Unique IDs: {unique_ids}")
            lines.append(f"  Pruned: {'yes' if unique_ids < self.count_leaves() else 'no'}")
        lines.append("")
        
        if not self.root:
            lines.append("  (empty tree)")
        else:
            self.__format_tree(self.root, "", "", lines)
        
        return "\n".join(lines)
    
    def __format_tree(self, node: Dict[str, Any], prefix: str, path: str, lines: List[str]) -> None:
        """
        Recursively format tree structure.
        
        Args:
            node (Dict[str, Any]): Current subtree
            prefix (str): Line prefix for indentation
            path (str): Current path
            lines (List[str]): Output lines to append to
        """
        items = list(node.items())
        for i, (name, child) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "`-- " if is_last else "|-- "
            child_path = f"{path}.{name}" if path else name
            
            if self.__built and child_path in self.path_to_id:
                loc_id = self.path_to_id[child_path]
                lines.append(f"{prefix}{connector}{name} [ID={loc_id}]")
            else:
                lines.append(f"{prefix}{connector}{name}")
            
            if child:
                child_prefix = prefix + ("    " if is_last else "|   ")
                self.__format_tree(child, child_prefix, child_path, lines)
    
    def print_tree(self) -> None:
        """Print the tree structure to stdout."""
        print(str(self))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export tree as a nested dictionary.
        
        Returns:
            (Dict[str, Any]): Tree structure with IDs
        """
        def add_ids(node: Dict, path: str) -> Dict:
            result = {}
            for name, child in node.items():
                child_path = f"{path}.{name}" if path else name
                if child:
                    result[name] = add_ids(child, child_path)
                else:
                    loc_id = self.path_to_id.get(child_path, -1) if self.__built else -1
                    result[name] = {"__id__": loc_id}
            return result
        
        return add_ids(self.root, "")


class WeightedLocationTree(LocationTree):
    """
    Location tree with importance weights for selective pruning.
    
    Higher-weight paths are preserved at deeper levels while
    lower-weight paths are collapsed earlier.
    
    Usage:
        tree = WeightedLocationTree(max_locations=1024)
        tree.set_weight("head.*", 10.0)       # Preserve classifier granularity
        tree.set_weight("encoder.layer*", 1.0) # Collapse repeated layers
        
        for path in module_paths:
            tree.add_path(path)
        
        tree.build_ids()
    
    Attributes:
        weights (Dict[str, float]): Pattern -> weight mapping
    """
    
    def __init__(self, max_locations: int = 1024) -> None:
        """
        Initialize the weighted tree.
        
        Args:
            max_locations (int): Maximum number of location IDs
        """
        super().__init__(max_locations)
        self.weights: Dict[str, float] = {}
        self.__default_weight: float = 1.0
    
    def set_weight(self, path_pattern: str, weight: float) -> None:
        """
        Set importance weight for paths matching pattern.
        
        Higher weights mean more granularity preserved.
        
        Args:
            path_pattern (str): Path pattern (supports * wildcard at end)
            weight (float): Importance weight (higher = more important)
        """
        self.weights[path_pattern] = weight
    
    def set_default_weight(self, weight: float) -> None:
        """
        Set default weight for unmatched paths.
        
        Args:
            weight (float): Default weight value
        """
        self.__default_weight = weight
    
    def __get_weight(self, path: str) -> float:
        """
        Get weight for a path.
        
        Checks patterns in order, returns first match or default.
        
        Args:
            path (str): Module path
            
        Returns:
            (float): Weight value
        """
        for pattern, weight in self.weights.items():
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if path.startswith(prefix):
                    return weight
            elif path == pattern:
                return weight
        return self.__default_weight
    
    def __repr__(self) -> str:
        """Short string representation."""
        base = super().__repr__()
        return base.replace("LocationTree", "WeightedLocationTree")
