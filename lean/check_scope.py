"""Check the current proof source set and axiom-audit import closure."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
sources = sorted(ROOT.glob('LearnedTG*.lean'))
modules = {p.stem for p in sources}
roots = set(re.findall(r'`(LearnedTG\w+)', (ROOT / 'lakefile.lean').read_text()))
assert modules == roots, 'The source modules and build roots differ.'
graph = {}
for path in sources:
    text = path.read_text()
    assert not re.search(r'^\s*(?:axiom|sorry|admit)\b', text, re.M), path.name
    graph[path.stem] = set(re.findall(r'^(?:public )?import (LearnedTG\w+)', text, re.M))
    assert graph[path.stem] <= modules, path.name
pending, audited = ['LearnedTGAxiomAudit'], set()
while pending:
    name = pending.pop()
    if name not in audited:
        audited.add(name)
        pending.extend(graph[name])
assert audited == modules, 'A source module is missing from the axiom audit.'
assert len(modules) == 16, 'Unexpected formalization scope.'
print(f'Proof scope passed: {len(modules)} modules; every module enters the axiom audit.')
