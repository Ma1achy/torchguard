"""
Tests for error flags features.

Tests cover:
- FX-based location extraction (extract_locations, location_from_stack)
- LocationTree (basic operations, tree building, prefix matching)
- LocationTree pruning (auto-collapsing when >max_locations)
- WeightedLocationTree (weight patterns, default weights)
- Accumulation modes (CHRONOLOGICAL, CHRONOLOGICAL_FIRST, SEVERITY, LOCATION)

Run with:
    pytest tests/utils/errors/compiled/test_location.py -v
"""
import pytest
import torch
import torch.nn as nn
from typing import Dict

from torchguard import (
    ErrorCode,
    ErrorLocation,
    Severity,
    AccumulationConfig,
    Priority,
    Order,
    Dedupe,
    ErrorConfig,
    CONFIG,
    error_t,
    err,
    flags as flags_ns,
)
# Internal location utilities for testing
from ..src.core.location import (
    LocationTree,
    WeightedLocationTree,
    extract_locations,
    location_from_stack,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ERRORCODE DEFAULT SEVERITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorCodeDefaultSeverity:
    """Tests for ErrorCode.default_severity() method."""
    
    def test_default_severity_critical(self) -> None:
        """NaN and Inf should be CRITICAL."""
        assert ErrorCode.default_severity(ErrorCode.NAN) == Severity.CRITICAL
        assert ErrorCode.default_severity(ErrorCode.INF) == Severity.CRITICAL
    
    def test_default_severity_error(self) -> None:
        """Index and overflow issues should be ERROR."""
        assert ErrorCode.default_severity(ErrorCode.OUT_OF_BOUNDS) == Severity.ERROR
        assert ErrorCode.default_severity(ErrorCode.NEGATIVE_IDX) == Severity.ERROR
        assert ErrorCode.default_severity(ErrorCode.OVERFLOW) == Severity.ERROR
        assert ErrorCode.default_severity(ErrorCode.EMPTY_INPUT) == Severity.ERROR
    
    def test_default_severity_warn(self) -> None:
        """Quality and runtime issues should be WARN."""
        assert ErrorCode.default_severity(ErrorCode.ZERO_OUTPUT) == Severity.WARN
        assert ErrorCode.default_severity(ErrorCode.CONSTANT_OUTPUT) == Severity.WARN
        assert ErrorCode.default_severity(ErrorCode.SATURATED) == Severity.WARN
        assert ErrorCode.default_severity(ErrorCode.FALLBACK_VALUE) == Severity.WARN
        assert ErrorCode.default_severity(ErrorCode.VALUE_CLAMPED) == Severity.WARN
    
    def test_default_severity_ok(self) -> None:
        """OK code should return OK severity."""
        assert ErrorCode.default_severity(ErrorCode.OK) == Severity.OK
    
    def test_default_severity_unknown(self) -> None:
        """Unknown codes should default to ERROR."""
        assert ErrorCode.default_severity(99) == Severity.ERROR
    
    def test_default_severity_aliases(self) -> None:
        """
        Verify compatibility aliases map to same severity as underlying codes.
        
        Expected: Each alias returns same severity as its underlying code.
        """
        # UNDERFLOW -> OVERFLOW -> ERROR
        assert ErrorCode.default_severity(ErrorCode.UNDERFLOW) == Severity.ERROR
        # INVALID_VALUE -> NEGATIVE_IDX -> ERROR
        assert ErrorCode.default_severity(ErrorCode.INVALID_VALUE) == Severity.ERROR
        # UNSTABLE -> SATURATED -> WARN
        assert ErrorCode.default_severity(ErrorCode.UNSTABLE) == Severity.WARN
        # CLAMPED -> VALUE_CLAMPED -> WARN
        assert ErrorCode.default_severity(ErrorCode.CLAMPED) == Severity.WARN
        # SPARSE -> ZERO_OUTPUT -> WARN
        assert ErrorCode.default_severity(ErrorCode.SPARSE) == Severity.WARN


# ═══════════════════════════════════════════════════════════════════════════════
# FX EXTRACTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFXExtraction:
    """Tests for FX-based auto-location extraction."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_extract_locations_injects_fx_path(self) -> None:
        """Verify _fx_path is injected into all submodules."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(8, 4)
                self.decoder = nn.Linear(4, 8)
            
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
        model = Model()
        locations = extract_locations(model)
        
        # Check _fx_path injection
        assert hasattr(model.encoder, '_fx_path')
        assert model.encoder._fx_path == "encoder"
        assert model.decoder._fx_path == "decoder"
        
        # Check locations returned
        assert "encoder" in locations
        assert "decoder" in locations
    
    def test_extract_locations_without_example_input(self) -> None:
        """Verify extract_locations works without example_input."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(8, 4)
            
            def forward(self, x):
                return self.layer(x)
        
        model = Model()
        
        # Should not raise, should return locations
        locations = extract_locations(model)  # No example_input
        
        assert len(locations) > 0
        assert "layer" in locations
    
    def test_extract_locations_nested_modules_fx_path(self) -> None:
        """Verify nested module paths are correct."""
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 4)
            
            def forward(self, x):
                return self.proj(x)
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
            
            def forward(self, x):
                return self.encoder(x)
        
        model = Model()
        locations = extract_locations(model)
        
        # Check nested path
        assert hasattr(model.encoder.proj, '_fx_path')
        assert model.encoder.proj._fx_path == "encoder.proj"
        assert "encoder.proj" in locations
    
    def test_extract_locations_simple_model(self) -> None:
        """extract_locations should register all submodule paths."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(20, 5)
            
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        model = SimpleModel()
        example_input = torch.randn(2, 10)
        
        locations = extract_locations(model, example_input)
        
        # Should have registered all submodules
        assert "linear1" in locations
        assert "relu" in locations
        assert "linear2" in locations
        
        # All should have unique IDs >= 16 (after built-ins)
        assert locations["linear1"] >= 1  # After UNKNOWN (0)
        assert locations["relu"] >= 1  # After UNKNOWN (0)
        assert locations["linear2"] >= 1  # After UNKNOWN (0)
        assert len(set(locations.values())) == 3  # All unique
    
    def test_extract_locations_nested_model(self) -> None:
        """extract_locations should handle nested modules."""
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(10, 20, 3, padding=1)
                self.norm = nn.BatchNorm1d(20)
            
            def forward(self, x):
                return self.norm(self.conv(x))
        
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.decoder = nn.Linear(20, 10)
            
            def forward(self, x):
                # x: (batch, 10, seq)
                x = self.encoder(x)
                x = x.mean(dim=-1)  # Global pooling
                return self.decoder(x)
        
        model = NestedModel()
        example_input = torch.randn(2, 10, 5)
        
        locations = extract_locations(model, example_input)
        
        # Should have nested paths
        assert "encoder" in locations or "encoder.conv" in locations
        assert "decoder" in locations
    
    def test_extract_locations_sequential(self) -> None:
        """extract_locations should handle Sequential containers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        example_input = torch.randn(2, 10)
        
        locations = extract_locations(model, example_input)
        
        # Sequential uses numeric indices
        assert "0" in locations
        assert "1" in locations
        assert "2" in locations
    
    def test_extract_locations_idempotent(self) -> None:
        """Calling extract_locations twice should return same IDs."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        example_input = torch.randn(2, 10)
        
        locations1 = extract_locations(model, example_input)
        locations2 = extract_locations(model, example_input)
        
        assert locations1 == locations2


class TestLocationFromStack:
    """Tests for location_from_stack function."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_empty_stack_returns_unknown(self) -> None:
        """Empty nn_module_stack should return UNKNOWN."""
        result = location_from_stack({})
        assert result == ErrorLocation.UNKNOWN
    
    def test_single_entry_stack(self) -> None:
        """Single entry stack should register that path."""
        stack = {"model": ("Model", type)}
        result = location_from_stack(stack)
        
        assert result >= 1  # Auto-registered (after UNKNOWN=0)
        assert ErrorLocation.name(result) == "model"
    
    def test_nested_stack_returns_deepest(self) -> None:
        """Nested stack should return deepest (last) path."""
        stack = {
            "model": ("Model", type),
            "model.encoder": ("Encoder", type),
            "model.encoder.conv": ("Conv2d", type),
        }
        result = location_from_stack(stack)
        
        # Should be the deepest path
        assert ErrorLocation.name(result) == "model.encoder.conv"
    
    def test_same_stack_returns_same_id(self) -> None:
        """Same stack should return same ID (idempotent)."""
        stack = {"model.layer": ("Linear", type)}
        
        id1 = location_from_stack(stack)
        id2 = location_from_stack(stack)
        
        assert id1 == id2


# ═══════════════════════════════════════════════════════════════════════════════
# LOCATION TREE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocationTree:
    """Tests for LocationTree basic operations."""
    
    def test_add_path_basic(self) -> None:
        """add_path should build tree structure."""
        tree = LocationTree(max_locations=100)
        tree.add_path("encoder.layer1.conv")
        tree.add_path("encoder.layer1.norm")
        tree.add_path("encoder.layer2.conv")
        
        # Check tree structure
        assert "encoder" in tree.root
        assert "layer1" in tree.root["encoder"]
        assert "layer2" in tree.root["encoder"]
        assert "conv" in tree.root["encoder"]["layer1"]
        assert "norm" in tree.root["encoder"]["layer1"]
    
    def test_count_leaves(self) -> None:
        """count_leaves should count leaf nodes."""
        tree = LocationTree(max_locations=100)
        tree.add_path("a.b.c")
        tree.add_path("a.b.d")
        tree.add_path("a.e")
        
        # Leaves: c, d, e (3 total)
        assert tree.count_leaves() == 3
    
    def test_build_ids_no_pruning(self) -> None:
        """build_ids should assign unique IDs when under budget."""
        tree = LocationTree(max_locations=100)
        tree.add_path("encoder.conv")
        tree.add_path("encoder.norm")
        tree.add_path("decoder.linear")
        
        tree.build_ids(start_id=100)
        
        # All leaves should have unique IDs
        assert "encoder.conv" in tree.path_to_id
        assert "encoder.norm" in tree.path_to_id
        assert "decoder.linear" in tree.path_to_id
        
        ids = list(tree.path_to_id.values())
        assert len(ids) == len(set(ids))  # All unique
        assert all(id >= 100 for id in ids)  # Start from 100
    
    def test_get_location_direct_match(self) -> None:
        """get_location should return ID for exact path match."""
        tree = LocationTree(max_locations=100)
        tree.add_path("encoder.conv")
        tree.build_ids(start_id=100)
        
        loc_id = tree.get_location("encoder.conv")
        assert loc_id == tree.path_to_id["encoder.conv"]
    
    def test_get_location_unknown_path(self) -> None:
        """get_location should return UNKNOWN for unmatched path."""
        tree = LocationTree(max_locations=100)
        tree.add_path("encoder.conv")
        tree.build_ids()
        
        loc_id = tree.get_location("decoder.linear")
        assert loc_id == ErrorLocation.UNKNOWN
    
    def test_cannot_add_after_build(self) -> None:
        """add_path should raise error after build_ids."""
        tree = LocationTree(max_locations=100)
        tree.add_path("encoder.conv")
        tree.build_ids()
        
        with pytest.raises(RuntimeError, match="Cannot add paths"):
            tree.add_path("decoder.linear")
    
    def test_must_build_before_get(self) -> None:
        """get_location should raise error before build_ids."""
        tree = LocationTree(max_locations=100)
        tree.add_path("encoder.conv")
        
        with pytest.raises(RuntimeError, match="Must call build_ids"):
            tree.get_location("encoder.conv")
    
    def test_register_all(self) -> None:
        """register_all should register all paths with global registry."""
        ErrorLocation.reset()
        
        tree = LocationTree(max_locations=100)
        tree.add_path("custom.path1")
        tree.add_path("custom.path2")
        tree.build_ids()
        
        registered = tree.register_all()
        
        # Should be registered with global registry
        assert ErrorLocation.get("custom.path1") != ErrorLocation.UNKNOWN
        assert ErrorLocation.get("custom.path2") != ErrorLocation.UNKNOWN


class TestLocationTreePruning:
    """Tests for LocationTree pruning behavior."""
    
    def test_pruning_when_over_budget(self) -> None:
        """Tree should prune when leaves exceed max_locations."""
        tree = LocationTree(max_locations=5)
        
        # Add 10 deep paths (more than max_locations)
        # Structure: encoder.layer{i}.attn.{q,k} = 20 leaves total
        for i in range(5):
            tree.add_path(f"encoder.layer{i}.attn.q")
            tree.add_path(f"encoder.layer{i}.attn.k")
        
        tree.build_ids()
        
        # Should have <= max_locations unique IDs due to pruning
        unique_ids = set(tree.path_to_id.values())
        assert len(unique_ids) <= 5
    
    def test_deep_tree_pruning(self) -> None:
        """Deep tree should be pruned at appropriate depth."""
        tree = LocationTree(max_locations=4)
        
        # Create deep tree with many leaves
        # encoder.layer0.attn.q, encoder.layer0.attn.k, encoder.layer0.attn.v
        # encoder.layer1.attn.q, encoder.layer1.attn.k, encoder.layer1.attn.v
        for layer in range(2):
            for proj in ["q", "k", "v"]:
                tree.add_path(f"encoder.layer{layer}.attn.{proj}")
        
        # 6 leaves, but only 4 allowed
        tree.build_ids()
        
        unique_ids = set(tree.path_to_id.values())
        assert len(unique_ids) <= 4
    
    def test_pruning_collapses_subtrees(self) -> None:
        """Pruning should collapse entire subtrees to single IDs."""
        tree = LocationTree(max_locations=2)
        
        # Two main branches, each with multiple leaves
        tree.add_path("encoder.conv1")
        tree.add_path("encoder.conv2")
        tree.add_path("decoder.linear1")
        tree.add_path("decoder.linear2")
        
        tree.build_ids()
        
        # With max 2, should collapse to 2 top-level: encoder, decoder
        unique_ids = set(tree.path_to_id.values())
        assert len(unique_ids) <= 2
        
        # All encoder paths should share same ID
        encoder_ids = {tree.path_to_id[p] for p in tree.path_to_id if p.startswith("encoder")}
        assert len(encoder_ids) == 1
    
    def test_longest_prefix_matching(self) -> None:
        """get_location should use longest prefix matching after pruning."""
        tree = LocationTree(max_locations=1)
        
        # All paths will collapse to single ID
        tree.add_path("encoder.layer0.conv")
        tree.add_path("encoder.layer1.conv")
        tree.build_ids()
        
        # Query for a path not explicitly stored
        loc_id = tree.get_location("encoder.layer0.conv.weight")
        
        # Should match the parent (encoder.layer0.conv or its collapsed parent)
        assert loc_id != ErrorLocation.UNKNOWN
    
    def test_descendants_share_id_after_pruning(self) -> None:
        """All descendants of pruned node should share same ID."""
        tree = LocationTree(max_locations=1)
        
        tree.add_path("encoder.layer0.attn.q")
        tree.add_path("encoder.layer0.attn.k")
        tree.add_path("encoder.layer0.ffn.linear1")
        tree.build_ids()
        
        # All should share same ID since max=1
        ids = set(tree.path_to_id.values())
        assert len(ids) == 1


class TestWeightedLocationTree:
    """Tests for WeightedLocationTree with importance weights."""
    
    def test_set_weight_exact_match(self) -> None:
        """set_weight should match exact path."""
        tree = WeightedLocationTree(max_locations=100)
        tree.set_weight("encoder.important", 10.0)
        tree.add_path("encoder.important")
        tree.add_path("encoder.regular")
        tree.build_ids()
        
        # Both paths should have IDs (no pruning needed at 100 max)
        assert "encoder.important" in tree.path_to_id
        assert "encoder.regular" in tree.path_to_id
    
    def test_set_weight_wildcard(self) -> None:
        """set_weight with * should match prefix."""
        tree = WeightedLocationTree(max_locations=100)
        tree.set_weight("head.*", 10.0)
        tree.add_path("head.linear1")
        tree.add_path("head.linear2")
        tree.add_path("encoder.conv")
        tree.build_ids()
        
        # All should get IDs
        assert "head.linear1" in tree.path_to_id
        assert "head.linear2" in tree.path_to_id
        assert "encoder.conv" in tree.path_to_id
    
    def test_set_default_weight(self) -> None:
        """set_default_weight should affect unmatched paths."""
        tree = WeightedLocationTree(max_locations=100)
        tree.set_default_weight(0.5)
        tree.add_path("encoder.conv")
        tree.build_ids()
        
        # Should still build successfully
        assert "encoder.conv" in tree.path_to_id
    
    def test_weights_inherit_from_location_tree(self) -> None:
        """WeightedLocationTree should have all LocationTree functionality."""
        tree = WeightedLocationTree(max_locations=100)
        tree.add_path("encoder.conv")
        tree.add_path("decoder.linear")
        
        # count_leaves should work
        assert tree.count_leaves() == 2
        
        tree.build_ids()
        
        # get_location should work
        loc = tree.get_location("encoder.conv")
        assert loc != ErrorLocation.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
# ACCUMULATION CONFIG TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccumulationChronoLast:
    """Tests for CHRONO+LAST accumulation (LIFO, keep newest)."""
    
    def test_newest_in_slot_0(self) -> None:
        """Most recent error should be in slot 0."""
        config = ErrorConfig(
            num_slots=8,
            accumulation=AccumulationConfig(priority=Priority.CHRONO, order=Order.LAST)
        )
        flags = err.new_t(1, config=config)
        
        # Push errors in sequence
        code1 = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.ERROR, config=config)
        
        code2 = torch.tensor([ErrorCode.INF], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.CRITICAL, config=config)
        
        code3 = torch.tensor([ErrorCode.OUT_OF_BOUNDS], dtype=torch.int64)
        flags = err.push(flags, code3, location=3, severity=Severity.WARN, config=config)
        
        # Slot 0 should have newest (OUT_OF_BOUNDS)
        first_code = err.get_first_code(flags).item()
        assert first_code == ErrorCode.OUT_OF_BOUNDS
    
    def test_oldest_dropped_when_full(self) -> None:
        """Oldest errors should drop when slots are full."""
        # num_slots=4 means num_words=1 (4 slots per word)
        # Push 5 errors to force oldest to drop
        config = ErrorConfig(
            num_slots=4,
            accumulation=AccumulationConfig(priority=Priority.CHRONO, order=Order.LAST)
        )
        flags = err.new_t(1, config=config)
        
        # Push 5 errors into 4 slots
        codes = [ErrorCode.NAN, ErrorCode.INF, ErrorCode.OVERFLOW, ErrorCode.OUT_OF_BOUNDS, ErrorCode.ZERO_OUTPUT]
        for i, code in enumerate(codes):
            c = torch.tensor([code], dtype=torch.int64)
            flags = err.push(flags, c, location=i+1, severity=Severity.ERROR, config=config)
        
        # Should only have 4 errors (newest four)
        errors = flags_ns.unpack(flags, 0, config)
        assert len(errors) == 4
        
        # Newest should be ZERO_OUTPUT (last pushed)
        assert errors[0].code == ErrorCode.ZERO_OUTPUT
        
        # Oldest (NAN) should have been dropped
        codes_found = [e.code for e in errors]
        assert ErrorCode.NAN not in codes_found


class TestAccumulationChronoFirst:
    """Tests for CHRONO+FIRST accumulation (FIFO, keep oldest/root cause)."""
    
    def test_oldest_in_slot_0(self) -> None:
        """First error should stay in slot 0."""
        config = ErrorConfig(
            num_slots=8,
            accumulation=AccumulationConfig(priority=Priority.CHRONO, order=Order.FIRST)
        )
        flags = err.new_t(1, config=config)
        
        # Push errors in sequence
        code1 = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.ERROR, config=config)
        
        code2 = torch.tensor([ErrorCode.INF], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.CRITICAL, config=config)
        
        # Slot 0 should have first error (NAN)
        first_code = err.get_first_code(flags).item()
        assert first_code == ErrorCode.NAN
    
    def test_new_errors_dropped_when_full(self) -> None:
        """New errors should be dropped when slots are full."""
        config = ErrorConfig(
            num_slots=2,
            accumulation=AccumulationConfig(priority=Priority.CHRONO, order=Order.FIRST)
        )
        flags = err.new_t(1, config=config)
        
        # Push 3 errors into 2 slots
        for i, code in enumerate([ErrorCode.NAN, ErrorCode.INF, ErrorCode.OVERFLOW]):
            c = torch.tensor([code], dtype=torch.int64)
            flags = err.push(flags, c, location=i+1, severity=Severity.ERROR, config=config)
        
        # Should only have 2 errors (oldest)
        errors = flags_ns.unpack(flags, 0, config)
        assert len(errors) == 2
        
        # First should be NAN (root cause preserved)
        assert errors[0].code == ErrorCode.NAN
        assert errors[1].code == ErrorCode.INF
        
        # OVERFLOW should NOT be present
        codes = [e.code for e in errors]
        assert ErrorCode.OVERFLOW not in codes
    
    def test_root_cause_preservation(self) -> None:
        """Root cause (first error) should always be preserved."""
        config = ErrorConfig(
            num_slots=1,
            accumulation=AccumulationConfig(priority=Priority.CHRONO, order=Order.FIRST)
        )
        flags = err.new_t(1, config=config)
        
        # Push multiple errors
        code1 = torch.tensor([ErrorCode.OUT_OF_BOUNDS], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.ERROR, config=config)
        
        code2 = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.CRITICAL, config=config)
        
        # Root cause preserved even with only 1 slot
        first_code = err.get_first_code(flags).item()
        assert first_code == ErrorCode.OUT_OF_BOUNDS


class TestAccumulationSeverityPriority:
    """Tests for SEVERITY+LAST accumulation (keep worst errors)."""
    
    def test_higher_severity_replaces_lower(self) -> None:
        """Higher severity errors should replace lower severity ones."""
        config = ErrorConfig(
            num_slots=2,
            accumulation=AccumulationConfig(priority=Priority.SEVERITY, order=Order.LAST)
        )
        flags = err.new_t(1, config=config)
        
        # Fill with WARN level errors
        code1 = torch.tensor([ErrorCode.ZERO_OUTPUT], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.WARN, config=config)
        
        code2 = torch.tensor([ErrorCode.FALLBACK_VALUE], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.WARN, config=config)
        
        # Push CRITICAL error - should replace one WARN
        code3 = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code3, location=3, severity=Severity.CRITICAL, config=config)
        
        # Should have CRITICAL error
        errors = flags_ns.unpack(flags, 0, config)
        severities = [e.severity for e in errors]
        assert Severity.CRITICAL in severities
    
    def test_lower_severity_uses_chronological(self) -> None:
        """Lower severity than all existing should use chronological fallback."""
        # num_slots=4 gives 1 word with 4 physical slots
        config = ErrorConfig(
            num_slots=4,
            accumulation=AccumulationConfig(priority=Priority.SEVERITY, order=Order.LAST)
        )
        flags = err.new_t(1, config=config)
        
        # Fill with CRITICAL errors
        code1 = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.CRITICAL, config=config)
        
        code2 = torch.tensor([ErrorCode.INF], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.CRITICAL, config=config)
        
        # Push WARN - can't replace CRITICAL, should use chronological fallback
        code3 = torch.tensor([ErrorCode.ZERO_OUTPUT], dtype=torch.int64)
        flags = err.push(flags, code3, location=3, severity=Severity.WARN, config=config)
        
        # All 3 errors should be present (WARN added via chronological since space available)
        errors = flags_ns.unpack(flags, 0, config)
        assert len(errors) == 3
        
        # Verify all severities present
        severities = [e.severity for e in errors]
        assert Severity.CRITICAL in severities
        assert Severity.WARN in severities


class TestAccumulationDedupeLocation:
    """Tests for Dedupe.LOCATION (dedupe by location)."""
    
    def test_same_location_updates_if_worse(self) -> None:
        """Same location error should update if worse."""
        config = ErrorConfig(
            num_slots=8,
            accumulation=AccumulationConfig(dedupe=Dedupe.LOCATION)
        )
        flags = err.new_t(1, config=config)
        
        # Push WARN at location 1
        code1 = torch.tensor([ErrorCode.ZERO_OUTPUT], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.WARN, config=config)
        
        # Push CRITICAL at same location 1 - should update
        code2 = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code2, location=1, severity=Severity.CRITICAL, config=config)
        
        errors = flags_ns.unpack(flags, 0, config)
        
        # Should only have 1 error (deduplicated)
        assert len(errors) == 1
        # Should be the worse one (CRITICAL)
        assert errors[0].severity == Severity.CRITICAL
    
    def test_different_locations_both_kept(self) -> None:
        """Different locations should both be kept."""
        config = ErrorConfig(
            num_slots=8,
            accumulation=AccumulationConfig(dedupe=Dedupe.LOCATION)
        )
        flags = err.new_t(1, config=config)
        
        # Push at location 1
        code1 = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.ERROR, config=config)
        
        # Push at location 2
        code2 = torch.tensor([ErrorCode.INF], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.ERROR, config=config)
        
        errors = flags_ns.unpack(flags, 0, config)
        
        # Both should be present
        assert len(errors) == 2
        locations = {e.location for e in errors}
        assert 1 in locations
        assert 2 in locations
    
    def test_same_location_keeps_better_if_new_is_worse(self) -> None:
        """Existing better error should NOT be replaced by worse."""
        config = ErrorConfig(
            num_slots=8,
            accumulation=AccumulationConfig(dedupe=Dedupe.LOCATION)
        )
        flags = err.new_t(1, config=config)
        
        # Push CRITICAL at location 1
        code1 = torch.tensor([ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code1, location=1, severity=Severity.CRITICAL, config=config)
        
        # Push WARN at same location 1 - should NOT replace
        code2 = torch.tensor([ErrorCode.ZERO_OUTPUT], dtype=torch.int64)
        flags = err.push(flags, code2, location=1, severity=Severity.WARN, config=config)
        
        errors = flags_ns.unpack(flags, 0, config)
        
        # Should still have CRITICAL
        assert len(errors) == 1
        assert errors[0].severity == Severity.CRITICAL
        assert errors[0].code == ErrorCode.NAN


class TestAccumulationConfigsCompiled:
    """Tests that accumulation configs work with torch.compile."""
    
    @pytest.mark.parametrize("acc_config", [
        AccumulationConfig(priority=Priority.CHRONO, order=Order.LAST),
        AccumulationConfig(priority=Priority.CHRONO, order=Order.FIRST),
        AccumulationConfig(priority=Priority.SEVERITY, order=Order.LAST),
        AccumulationConfig(dedupe=Dedupe.LOCATION),
    ])
    def test_config_compiles(self, acc_config: AccumulationConfig) -> None:
        """All accumulation configs should be compilable."""
        config = ErrorConfig(num_slots=4, accumulation=acc_config)
        
        @torch.compile(backend="eager", fullgraph=True)
        def push_error(flags, code, loc, sev):
            return err.push(flags, code, loc, sev, config)
        
        flags = err.new_t(3, config=config)
        code = torch.tensor([ErrorCode.NAN, ErrorCode.OK, ErrorCode.INF], dtype=torch.int64)
        
        # Should not raise
        result = push_error(flags, code, 1, Severity.ERROR)
        assert result.shape == (3, config.num_words)


class TestAccumulationConfigsBatchProcessing:
    """Tests for accumulation configs with batch processing."""
    
    def test_chrono_last_per_sample(self) -> None:
        """CHRONO+LAST should work per-sample."""
        config = ErrorConfig(
            num_slots=4,
            accumulation=AccumulationConfig(priority=Priority.CHRONO, order=Order.LAST)
        )
        flags = err.new_t(3, config=config)
        
        # Sample 0: NAN, Sample 1: OK, Sample 2: INF
        code = torch.tensor([ErrorCode.NAN, ErrorCode.OK, ErrorCode.INF], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.ERROR, config=config)
        
        # Sample 0: Add OOB
        code = torch.tensor([ErrorCode.OUT_OF_BOUNDS, ErrorCode.OK, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=2, severity=Severity.WARN, config=config)
        
        # Check each sample
        errors_0 = flags_ns.unpack(flags, 0, config)
        errors_1 = flags_ns.unpack(flags, 1, config)
        errors_2 = flags_ns.unpack(flags, 2, config)
        
        # Sample 0: 2 errors (OOB newest)
        assert len(errors_0) == 2
        assert errors_0[0].code == ErrorCode.OUT_OF_BOUNDS  # Newest in slot 0
        
        # Sample 1: no errors
        assert len(errors_1) == 0
        
        # Sample 2: 1 error
        assert len(errors_2) == 1
        assert errors_2[0].code == ErrorCode.INF
    
    def test_dedupe_location_per_sample(self) -> None:
        """Dedupe.LOCATION should dedupe per-sample independently."""
        config = ErrorConfig(
            num_slots=4,
            accumulation=AccumulationConfig(dedupe=Dedupe.LOCATION)
        )
        flags = err.new_t(2, config=config)
        
        # Both samples: error at location 1
        code = torch.tensor([ErrorCode.ZERO_OUTPUT, ErrorCode.ZERO_OUTPUT], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.WARN, config=config)
        
        # Sample 0 only: update location 1 to worse error
        code = torch.tensor([ErrorCode.NAN, ErrorCode.OK], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL, config=config)
        
        # Sample 0 should have CRITICAL, Sample 1 should have WARN
        errors_0 = flags_ns.unpack(flags, 0, config)
        errors_1 = flags_ns.unpack(flags, 1, config)
        
        assert len(errors_0) == 1
        assert errors_0[0].severity == Severity.CRITICAL
        
        assert len(errors_1) == 1
        assert errors_1[0].severity == Severity.WARN


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestV2FeaturesIntegration:
    """Integration tests combining multiple V2 features."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_fx_extraction_with_location_tree(self) -> None:
        """FX extraction + LocationTree should work together."""
        # Create model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.Linear(20, 5)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
        
        model = SimpleModel()
        example_input = torch.randn(2, 10)
        
        # Extract locations
        locations = extract_locations(model, example_input)
        
        # Build tree from extracted locations
        tree = LocationTree(max_locations=100)
        for path in locations.keys():
            tree.add_path(path)
        tree.build_ids()
        
        # Should be able to query
        for path in locations.keys():
            loc = tree.get_location(path)
            assert loc != ErrorLocation.UNKNOWN
    
    def test_accumulation_config_with_flag_helpers(self) -> None:
        """Accumulation configs should work with flag_nan/flag_inf."""
        from torchguard import flag_nan, flag_inf, err
        
        config = ErrorConfig(
            num_slots=4,
            accumulation=AccumulationConfig(priority=Priority.SEVERITY, order=Order.LAST)
        )
        
        # Create tensor with various issues
        tensor = torch.tensor([
            [1.0, 2.0],           # OK
            [float('nan'), 1.0],  # NaN (CRITICAL)
            [float('inf'), 2.0],  # Inf (CRITICAL)
        ])
        
        flags = err.new_t(3, tensor.device, config)
        flags = flag_nan(tensor, 1, flags, config)
        flags = flag_inf(tensor, 1, flags, config)
        
        # Check detection worked
        assert err.is_err(flags)[1].item()  # Sample 1 has error
        assert err.is_err(flags)[2].item()  # Sample 2 has error
        
        # Unpack and verify severity
        errors_1 = flags_ns.unpack(flags, 1, config)
        errors_2 = flags_ns.unpack(flags, 2, config)
        
        assert errors_1[0].severity == Severity.CRITICAL
        assert errors_2[0].severity == Severity.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZED METHODS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCountErrorsVectorized:
    """Tests for vectorized count_errors() method."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_count_errors_vectorized(self) -> None:
        """Verify count_errors() is vectorized and correct."""
        from torchguard import push
        
        flags = err.new_t(4, torch.device('cpu'))
        
        # Push errors: sample 0 gets 2 errors, sample 3 gets 1 error
        flags = push(flags, ErrorCode.NAN, "loc1", where=torch.tensor([True, False, False, True]))
        flags = push(flags, ErrorCode.INF, "loc2", where=torch.tensor([True, False, False, False]))
        
        counts = err.count_errors(flags)
        assert counts[0].item() == 2  # NAN + INF
        assert counts[1].item() == 0
        assert counts[2].item() == 0
        assert counts[3].item() == 1  # NAN only
    
    def test_count_errors_returns_int32(self) -> None:
        """count_errors should return int32 dtype."""
        flags = err.new_t(3)
        counts = err.count_errors(flags)
        assert counts.dtype == torch.int32


class TestMaxSeverityVectorized:
    """Tests for vectorized max_severity() method."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_max_severity_vectorized(self) -> None:
        """Verify max_severity() returns correct max per sample."""
        from torchguard import push
        
        flags = err.new_t(3, torch.device('cpu'))
        
        # Sample 0: WARN + ERROR -> max is ERROR
        flags = push(flags, ErrorCode.ZERO_OUTPUT, "loc1", 
                     where=torch.tensor([True, False, False]), severity=Severity.WARN)
        flags = push(flags, ErrorCode.OUT_OF_BOUNDS, "loc2", 
                     where=torch.tensor([True, False, False]), severity=Severity.ERROR)
        
        # Sample 1: CRITICAL only
        flags = push(flags, ErrorCode.NAN, "loc1", 
                     where=torch.tensor([False, True, False]), severity=Severity.CRITICAL)
        
        max_sev = err.max_severity(flags)
        assert max_sev[0].item() == Severity.ERROR
        assert max_sev[1].item() == Severity.CRITICAL
        assert max_sev[2].item() == Severity.OK


class TestHasDomainVectorized:
    """Tests for vectorized has_domain() method."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_has_domain_vectorized(self) -> None:
        """Verify has_domain() is vectorized and correct."""
        from torchguard import ErrorDomain
        from torchguard import push
        
        flags = err.new_t(3)
        
        # Sample 0: NUMERIC domain (NAN)
        # Sample 1: INDEX domain (OUT_OF_BOUNDS)
        # Sample 2: No error
        flags = push(flags, ErrorCode.NAN, "loc1", where=torch.tensor([True, False, False]))
        flags = push(flags, ErrorCode.OUT_OF_BOUNDS, "loc2", where=torch.tensor([False, True, False]))
        
        numeric_mask = err.has_domain(flags, ErrorDomain.NUMERIC)
        index_mask = err.has_domain(flags, ErrorDomain.INDEX)
        quality_mask = err.has_domain(flags, ErrorDomain.QUALITY)
        
        assert numeric_mask.tolist() == [True, False, False]
        assert index_mask.tolist() == [False, True, False]
        assert quality_mask.tolist() == [False, False, False]


class TestClearMethod:
    """Tests for clear() method."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_clear_removes_specific_code(self) -> None:
        """Verify clear() removes only specified code."""
        from torchguard import push, find
        
        flags = err.new_t(2, torch.device('cpu'))
        
        flags = push(flags, ErrorCode.NAN, "loc1", where=torch.tensor([True, True]))
        flags = push(flags, ErrorCode.INF, "loc2", where=torch.tensor([True, False]))
        
        # Clear NaN
        cleared = err.clear(flags, ErrorCode.NAN)
        
        # NaN should be gone
        assert find(ErrorCode.NAN, cleared).sum().item() == 0
        
        # INF should remain in sample 0
        assert find(ErrorCode.INF, cleared)[0].item() == True
        assert find(ErrorCode.INF, cleared)[1].item() == False
    
    def test_clear_preserves_other_codes(self) -> None:
        """clear() should not affect other error codes."""
        from torchguard import push, find
        
        flags = err.new_t(1)
        flags = push(flags, ErrorCode.NAN, "loc1")
        flags = push(flags, ErrorCode.INF, "loc2")
        flags = push(flags, ErrorCode.OUT_OF_BOUNDS, "loc3")
        
        # Clear only INF
        cleared = err.clear(flags, ErrorCode.INF)
        
        # NAN and OOB should remain
        assert find(ErrorCode.NAN, cleared)[0].item() == True
        assert find(ErrorCode.OUT_OF_BOUNDS, cleared)[0].item() == True
        # INF should be gone
        assert find(ErrorCode.INF, cleared)[0].item() == False
    
    def test_clear_in_compiled_model(self) -> None:
        """Verify clear() works with torch.compile(fullgraph=True)."""
        from torchguard import push, find
        
        @torch.compile(backend="eager", fullgraph=True)
        def model_fn(x):
            n = x.shape[0]
            flags = err.new_t(n, x.device)
            nan_mask = torch.isnan(x).view(n, -1).any(dim=-1)
            code_tensor = torch.where(nan_mask, ErrorCode.NAN, ErrorCode.OK)
            flags = err.push(flags, code_tensor, location=1, severity=Severity.CRITICAL)
            z = torch.where(nan_mask.unsqueeze(-1), 0.0, x)
            flags = err.clear(flags, ErrorCode.NAN)
            return z, flags
        
        x = torch.randn(2, 8)
        x[0, 0] = float('nan')
        
        z, flags = model_fn(x)
        assert not find(ErrorCode.NAN, flags).any()


class TestSummaryMethod:
    """Tests for summary() method."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_summary_aggregates_correctly(self) -> None:
        """summary() should aggregate by location and code."""
        from torchguard import push
        
        f = err.new_t(4)
        
        # 2 NAN at loc1, 1 INF at loc1, 1 NAN at loc2
        f = push(f, ErrorCode.NAN, "loc1", where=torch.tensor([True, True, False, False]))
        f = push(f, ErrorCode.INF, "loc1", where=torch.tensor([True, False, False, False]))
        f = push(f, ErrorCode.NAN, "loc2", where=torch.tensor([False, False, True, False]))
        
        summary = flags_ns.summary(f)
        
        assert "loc1" in summary
        assert "loc2" in summary
        assert summary["loc1"]["NAN"] == 2
        assert summary["loc1"]["INF"] == 1
        assert summary["loc2"]["NAN"] == 1
    
    def test_summary_empty_flags(self) -> None:
        """summary() should return empty dict for no errors."""
        f = err.new_t(10)
        summary = flags_ns.summary(f)
        assert summary == {}


class TestReprMethod:
    """Tests for repr() method."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_repr_no_errors(self) -> None:
        """repr() should show 'no errors' for clean flags."""
        f = err.new_t(5)
        result = flags_ns.repr(f)
        assert "5 samples" in result
        assert "no errors" in result
    
    def test_repr_with_errors(self) -> None:
        """repr() should show error summary."""
        from torchguard import push
        
        f = err.new_t(3)
        f = push(f, ErrorCode.NAN, "encoder.layer0", where=torch.tensor([True, True, False]))
        f = push(f, ErrorCode.INF, "decoder.linear", where=torch.tensor([True, False, False]))
        
        result = flags_ns.repr(f)
        assert "3 samples" in result
        assert "3 errors" in result
        assert "NAN" in result


class TestVectorizedMethodsCompiled:
    """Tests that vectorized methods work with torch.compile."""
    
    def setup_method(self) -> None:
        """Reset location registry before each test."""
        ErrorLocation.reset()
    
    def test_count_errors_compiles(self) -> None:
        """count_errors should be compilable."""
        @torch.compile(backend="eager", fullgraph=True)
        def count_fn(flags):
            return err.count_errors(flags)
        
        flags = err.new_t(3)
        code = torch.tensor([ErrorCode.NAN, ErrorCode.OK, ErrorCode.INF], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        result = count_fn(flags)
        assert result.tolist() == [1, 0, 1]
    
    def test_max_severity_compiles(self) -> None:
        """max_severity should be compilable."""
        @torch.compile(backend="eager", fullgraph=True)
        def max_sev_fn(flags):
            return err.max_severity(flags)
        
        flags = err.new_t(2)
        code = torch.tensor([ErrorCode.NAN, ErrorCode.ZERO_OUTPUT], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        # Push another with lower severity to sample 1
        code2 = torch.tensor([ErrorCode.OK, ErrorCode.FALLBACK_VALUE], dtype=torch.int64)
        flags = err.push(flags, code2, location=2, severity=Severity.WARN)
        
        result = max_sev_fn(flags)
        assert result[0].item() == Severity.CRITICAL
        # Sample 1 has CRITICAL from first push (ZERO_OUTPUT with CRITICAL severity)
        assert result[1].item() == Severity.CRITICAL
    
    def test_clear_compiles(self) -> None:
        """clear should be compilable."""
        @torch.compile(backend="eager", fullgraph=True)
        def clear_fn(flags):
            return err.clear(flags, ErrorCode.NAN)
        
        flags = err.new_t(2)
        code = torch.tensor([ErrorCode.NAN, ErrorCode.NAN], dtype=torch.int64)
        flags = err.push(flags, code, location=1, severity=Severity.CRITICAL)
        
        result = clear_fn(flags)
        assert (result == 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# ERRORCONFIG STRICT VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorConfigStrictValidation:
    """Tests for ErrorConfig.strict_validation field."""
    
    def test_strict_validation_default_false(self) -> None:
        """strict_validation should default to False."""
        config = ErrorConfig()
        assert config.strict_validation is False
    
    def test_strict_validation_can_be_set_true(self) -> None:
        """strict_validation can be set to True."""
        config = ErrorConfig(strict_validation=True)
        assert config.strict_validation is True
    
    def test_strict_validation_in_custom_config(self) -> None:
        """strict_validation works with other custom config values."""
        config = ErrorConfig(
            num_slots=16,
            accumulation=AccumulationConfig(priority=Priority.SEVERITY, order=Order.LAST),
            strict_validation=True,
        )
        assert config.num_slots == 16
        assert config.accumulation.priority == Priority.SEVERITY
        assert config.strict_validation is True
    
    def test_default_config_has_strict_false(self) -> None:
        """Global CONFIG should have strict_validation=False by default."""
        assert CONFIG.strict_validation is False
    
    def test_config_is_mutable(self) -> None:
        """Config should be mutable to allow user customization."""
        config = ErrorConfig(strict_validation=True, flag_dtype=torch.int64)
        # Should be able to modify it
        config.strict_validation = False
        assert config.strict_validation is False

