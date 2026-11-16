"""Reject obvious legacy case dispatch before a new package POST.

This conservative source check implements the user's 2026-11-04 requirement.
It is a backstop, not a proof that every possible selector is appropriate.
"""
import re
from pathlib import Path


def violations(source):
    code = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
    problems = []
    if 'gmm_diagnostic_nan' in code:
        problems.append('diagnostic NaN mask requires explicit --diagnostic-probe mode')
    if re.search(r'\bis[ICR]\d+(?:Exact|Pad|Auto|[A-Za-z_]*)\b', code):
        problems.append('legacy case-named dispatch flags')
    if re.search(r'\bgrouped_matmul_gelu_(?:regular_[cr]\d+|[ir]\d+)[A-Za-z0-9_]*\b', code):
        problems.append('case-specific kernel entry points')
    if re.search(r'constexpr\s+int64_t\s+m\s*=\s*\d+\s*,\s*k\s*=\s*\d+\s*,\s*n\s*=\s*\d+', code):
        problems.append('complete input dimensions baked into a kernel')
    for condition in re.findall(r'\bif\s*\(([^{};]*?)\)\s*\{', code):
        # Tile members such as selectedTile.m describe resource dimensions;
        # they are not comparisons against the input M/K/N/E variables.
        matches = re.findall(r'(?<![.\w>])([mkne])\s*==\s*(\d+)', condition)
        dimensions = {name for name, value in matches if int(value) > 1}
        if len(dimensions) >= 3:
            problems.append('literal M/K/N/E tuple match in a condition')
            break
    return problems


def validate(source):
    problems = violations(source)
    if problems:
        raise RuntimeError(
            'Latest user requirement: future packages must use natural generic dispatch. '
            'No POST was sent. Source check rejected: ' + '; '.join(problems))


def validate_package(directory, diagnostic=False):
    if diagnostic:
        import diagnostic_probe
        return diagnostic_probe.validate(directory)
    validate((Path(directory) / 'kernel.asc').read_text())
    return None
