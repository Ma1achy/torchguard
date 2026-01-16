#!/usr/bin/env python3
"""Validate torchguard package structure before distribution.

Checks:
1. All intended exports are present
2. No circular imports
3. Required files exist (py.typed, MANIFEST.in)
4. Class methods are accessible
5. Package imports without errors

Run: python scripts/validate_package.py
"""

import sys
from pathlib import Path


def check_files_exist() -> list:
    """Check required files exist."""
    issues = []
    root = Path(__file__).parent.parent
    
    required_files = [
        ('torchguard/py.typed', 'PEP 561 type marker'),
        ('MANIFEST.in', 'Package distribution manifest'),
        ('README.md', 'Documentation'),
    ]
    
    for filepath, description in required_files:
        full_path = root / filepath
        if not full_path.exists():
            issues.append(f"Missing {description}: {filepath}")
    
    return issues


def check_primary_exports() -> list:
    """Check primary API exports."""
    issues = []
    
    try:
        import torchguard as tg
    except ImportError as e:
        return [f"Cannot import torchguard: {e}"]
    
    primary = ['err', 'flags', 'error_t', 'tracked', 'tensorcheck']
    for name in primary:
        if not hasattr(tg, name):
            issues.append(f"Missing primary export: {name}")
    
    return issues


def check_secondary_exports() -> list:
    """Check secondary/advanced API exports."""
    issues = []
    
    try:
        import torchguard as tg
    except ImportError as e:
        return [f"Cannot import torchguard: {e}"]
    
    secondary = [
        # Core types
        'ErrorCode', 'ErrorDomain', 'Severity',
        # Config
        'ErrorConfig', 'CONFIG', 'get_config', 'set_config',
        'AccumulationConfig', 'Priority', 'Order', 'Dedupe',
        # Location
        'ErrorLocation',
        # DSL
        'IF', 'HAS', 'IS', 'OR', 'AND', 'NOT',
        # Result
        'Ok', 'Err', 'Result',
        # Flags inspection
        'UnpackedError', 'ErrorFlags',
        # Helpers
        'push', 'find', 'fix', 'flag_nan', 'flag_inf', 'flag_oob_indices', 'has_err',
        # Decorators
        'as_result', 'as_exception', 'unwrap',
        # Validation errors
        'ValidationError', 'DimensionMismatchError', 'DTypeMismatchError',
        'DeviceMismatchError', 'InvalidParameterError', 'TypeMismatchError',
        'InvalidReturnTypeError',
        # Constants
        'SLOT_BITS', 'SLOTS_PER_WORD',
        'CODE_SHIFT', 'CODE_BITS', 'CODE_MASK',
        'LOCATION_SHIFT', 'LOCATION_BITS', 'LOCATION_MASK',
        'SEVERITY_SHIFT', 'SEVERITY_BITS', 'SEVERITY_MASK', 'SLOT_MASK',
        # Experimental
        'experimental',
        # Version
        '__version__',
    ]
    
    for name in secondary:
        if not hasattr(tg, name):
            issues.append(f"Missing secondary export: {name}")
    
    return issues


def check_class_methods() -> list:
    """Check class methods are accessible."""
    issues = []
    
    try:
        import torchguard as tg
    except ImportError as e:
        return [f"Cannot import torchguard: {e}"]
    
    # ErrorCode methods
    ec_methods = ['name', 'domain', 'subcode', 'is_critical', 'in_domain', 
                  'domain_name', 'default_severity']
    for method in ec_methods:
        if not hasattr(tg.ErrorCode, method):
            issues.append(f"Missing ErrorCode.{method}()")
        elif not callable(getattr(tg.ErrorCode, method)):
            issues.append(f"ErrorCode.{method} is not callable")
    
    # Severity methods
    sev_methods = ['name', 'is_critical', 'is_error_or_worse', 'is_warn_or_worse']
    for method in sev_methods:
        if not hasattr(tg.Severity, method):
            issues.append(f"Missing Severity.{method}()")
        elif not callable(getattr(tg.Severity, method)):
            issues.append(f"Severity.{method} is not callable")
    
    # ErrorDomain methods
    if not hasattr(tg.ErrorDomain, 'name'):
        issues.append("Missing ErrorDomain.name()")
    
    return issues


def check_class_methods_work() -> list:
    """Check class methods actually work."""
    issues = []
    
    try:
        import torchguard as tg
    except ImportError as e:
        return [f"Cannot import torchguard: {e}"]
    
    # Test ErrorCode.name()
    try:
        result = tg.ErrorCode.name(1)
        if result != "NAN":
            issues.append(f"ErrorCode.name(1) returned {result!r}, expected 'NAN'")
    except Exception as e:
        issues.append(f"ErrorCode.name() raised: {e}")
    
    # Test ErrorCode.is_critical()
    try:
        result = tg.ErrorCode.is_critical(1)
        if result != True:
            issues.append(f"ErrorCode.is_critical(1) returned {result}, expected True")
    except Exception as e:
        issues.append(f"ErrorCode.is_critical() raised: {e}")
    
    # Test Severity.name()
    try:
        result = tg.Severity.name(3)
        if result != "CRITICAL":
            issues.append(f"Severity.name(3) returned {result!r}, expected 'CRITICAL'")
    except Exception as e:
        issues.append(f"Severity.name() raised: {e}")
    
    # Test ErrorDomain.name()
    try:
        result = tg.ErrorDomain.name(0)
        if result != "NUMERIC":
            issues.append(f"ErrorDomain.name(0) returned {result!r}, expected 'NUMERIC'")
    except Exception as e:
        issues.append(f"ErrorDomain.name() raised: {e}")
    
    return issues


def check_circular_imports() -> list:
    """Check for circular import issues."""
    issues = []
    
    try:
        # Clear any cached imports
        import sys as _sys
        to_remove = [k for k in _sys.modules.keys() if k.startswith('torchguard')]
        for k in to_remove:
            del _sys.modules[k]
        
        # Import in various orders to detect circular dependencies
        import torchguard
        import torchguard.src.core
        import torchguard.src.err
        import torchguard.src.experimental
        import torchguard.src.decorators
        import torchguard.src.control
        
    except ImportError as e:
        issues.append(f"Circular import detected: {e}")
    
    return issues


def check_err_namespace() -> list:
    """Check err namespace has all expected methods."""
    issues = []
    
    try:
        import torchguard as tg
    except ImportError as e:
        return [f"Cannot import torchguard: {e}"]
    
    err_methods = [
        # Creation
        'new', 'new_t', 'from_code',
        # Modification
        'push',
        # Queries
        'is_ok', 'is_err', 'has_any', 'has_nan', 'has_inf', 'has_code',
        'has_critical', 'count_errors', 'max_severity',
        # Extraction
        'get_ok', 'get_err',
        # Error codes as attributes
        'OK', 'NAN', 'INF', 'OVERFLOW', 'OUT_OF_BOUNDS',
    ]
    
    for method in err_methods:
        if not hasattr(tg.err, method):
            issues.append(f"Missing err.{method}")
    
    # Also check top-level helper functions
    top_level_helpers = ['find', 'fix', 'push', 'flag_nan', 'flag_inf']
    for helper in top_level_helpers:
        if not hasattr(tg, helper):
            issues.append(f"Missing top-level helper: {helper}")
    
    return issues


def check_all_declaration() -> list:
    """Check __all__ is properly defined."""
    issues = []
    
    try:
        import torchguard as tg
    except ImportError as e:
        return [f"Cannot import torchguard: {e}"]
    
    if not hasattr(tg, '__all__'):
        issues.append("Missing __all__ declaration in top-level __init__.py")
        return issues
    
    # Check __all__ contains important exports
    important = ['err', 'flags', 'ErrorCode', 'CONFIG', 'get_config', 'set_config']
    for name in important:
        if name not in tg.__all__:
            issues.append(f"'{name}' not in __all__")
    
    return issues


def main():
    """Main entry point."""
    print("=" * 60)
    print("TORCHGUARD PACKAGE VALIDATION")
    print("=" * 60)
    
    all_issues = []
    
    # 1. Check files exist
    print("\n1. Checking required files...")
    issues = check_files_exist()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ All required files present")
    
    # 2. Check circular imports
    print("\n2. Checking for circular imports...")
    issues = check_circular_imports()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ No circular imports detected")
    
    # 3. Check primary exports
    print("\n3. Checking primary exports...")
    issues = check_primary_exports()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ All primary exports present")
    
    # 4. Check secondary exports
    print("\n4. Checking secondary exports...")
    issues = check_secondary_exports()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ All secondary exports present")
    
    # 5. Check class methods
    print("\n5. Checking class methods are accessible...")
    issues = check_class_methods()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ All class methods accessible")
    
    # 6. Check class methods work
    print("\n6. Testing class methods work correctly...")
    issues = check_class_methods_work()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ All class methods work correctly")
    
    # 7. Check err namespace
    print("\n7. Checking err namespace...")
    issues = check_err_namespace()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ err namespace complete")
    
    # 8. Check __all__ declaration
    print("\n8. Checking __all__ declaration...")
    issues = check_all_declaration()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ __all__ declaration complete")
    
    # Summary
    print("\n" + "=" * 60)
    if all_issues:
        print(f"❌ VALIDATION FAILED - {len(all_issues)} issue(s) found")
        sys.exit(1)
    else:
        print("✅ VALIDATION PASSED - Package ready for distribution")
        sys.exit(0)


if __name__ == '__main__':
    main()
