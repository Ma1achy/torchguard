#!/usr/bin/env python3
"""Check import order and organization in all Python files.

Import order should be:
1. __future__ imports (always first)
2. Standard library
3. Third-party (torch, packaging, etc.)
4. TorchGuard internal (relative imports)

Run: python scripts/check_imports.py
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


# Standard library modules (common ones)
STDLIB_PREFIXES = {
    'abc', 'asyncio', 'collections', 'contextlib', 'copy', 'dataclasses',
    'datetime', 'enum', 'functools', 'hashlib', 'importlib', 'inspect',
    'io', 'itertools', 'json', 'logging', 'math', 'os', 'pathlib',
    'pickle', 're', 'shutil', 'signal', 'socket', 'subprocess', 'sys',
    'tempfile', 'textwrap', 'threading', 'time', 'traceback', 'types',
    'typing', 'unittest', 'uuid', 'warnings', 'weakref',
}

# Third-party packages
THIRD_PARTY_PREFIXES = {'torch', 'packaging', 'numpy', 'pytest'}


def categorize_import(line: str) -> str:
    """Categorize an import line."""
    line = line.strip()
    
    if line.startswith('from __future__'):
        return 'future'
    
    if line.startswith('from .') or line.startswith('from ..'):
        return 'internal'
    
    # Extract module name
    if line.startswith('from '):
        match = re.match(r'from\s+(\w+)', line)
        module = match.group(1) if match else ''
    elif line.startswith('import '):
        match = re.match(r'import\s+(\w+)', line)
        module = match.group(1) if match else ''
    else:
        return 'unknown'
    
    if module in THIRD_PARTY_PREFIXES:
        return 'third_party'
    if module in STDLIB_PREFIXES:
        return 'stdlib'
    
    # Default: treat unknown as stdlib
    return 'stdlib'


def extract_imports(content: str) -> List[Tuple[int, str, str]]:
    """Extract all import lines with their line numbers and categories."""
    imports = []
    for i, line in enumerate(content.split('\n'), 1):
        line_stripped = line.strip()
        if line_stripped.startswith('from ') or line_stripped.startswith('import '):
            category = categorize_import(line_stripped)
            imports.append((i, line_stripped, category))
    return imports


def check_import_order(imports: List[Tuple[int, str, str]]) -> List[str]:
    """Check if imports are in correct order."""
    issues = []
    
    # Expected order
    order = ['future', 'stdlib', 'third_party', 'internal']
    current_order_idx = 0
    
    for line_no, line, category in imports:
        if category == 'unknown':
            continue
        
        try:
            cat_idx = order.index(category)
        except ValueError:
            continue
        
        if cat_idx < current_order_idx:
            issues.append(
                f"  Line {line_no}: {category} import should come before {order[current_order_idx]} imports"
            )
        else:
            current_order_idx = cat_idx
    
    return issues


def check_file(filepath: Path) -> List[str]:
    """Check import order in a file. Returns list of issues."""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [f"  Error reading file: {e}"]
    
    imports = extract_imports(content)
    if not imports:
        return []  # No imports, no issues
    
    return check_import_order(imports)


def main():
    """Main entry point."""
    root = Path(__file__).parent.parent / 'torchguard' / 'src'
    
    if not root.exists():
        print(f"Error: Source directory not found: {root}")
        sys.exit(1)
    
    print("=" * 60)
    print("TORCHGUARD IMPORT ORDER CHECK")
    print("=" * 60)
    print(f"\nScanning: {root}\n")
    
    issues_found = False
    files_checked = 0
    
    for pyfile in sorted(root.rglob('*.py')):
        if '__pycache__' in str(pyfile):
            continue
        
        files_checked += 1
        issues = check_file(pyfile)
        
        if issues:
            rel_path = pyfile.relative_to(root)
            print(f"❌ {rel_path}:")
            for issue in issues:
                print(issue)
            print()
            issues_found = True
    
    print("-" * 60)
    print(f"Files checked: {files_checked}")
    
    if issues_found:
        print("\n❌ IMPORT ORDER ISSUES FOUND")
        print("   Fix: Move imports to correct order: future → stdlib → third_party → internal")
        sys.exit(1)
    else:
        print("\n✅ All imports properly ordered")
        sys.exit(0)


if __name__ == '__main__':
    main()
