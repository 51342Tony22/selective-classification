# Snapshot validation

Date: 2026-11-01.

Result: 47 tests passed; one existing MLP batch-size warning on a tiny test fixture.

```bash
PYTHONPATH=src python -m pytest tests -q
```

Checks used the existing local project dependency environment against this upload copy. Dependencies were not freshly installed from scratch for this packaging check.

All Python source files in the new repository snapshots passed an AST syntax check. The selected upload files, including DOCX XML, were scanned for common credential-token and private-key patterns; no matching credentials were found. Pattern checks are limited and are not a comprehensive security audit. No model training, paid teacher requests, or full benchmark reruns were performed during packaging. Saved metrics describe historical experiments.

The separately retained `extensions/aps-bee-gate` revision also passed its complete existing test suite (51 tests) using `PYTHONPATH=src python -m pytest tests -q` from that directory. One existing small-fixture MLP batch-size warning remains. No APS training was launched.
