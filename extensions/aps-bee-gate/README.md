# APS and BEE gate extension

This later revision snapshot adds APS Failure data support and a validation-only boundary-error-enrichment (BEE) gate. It is retained separately from the main experiment implementation so the earlier saved result bundle remains tied to its own source version.

Run from this directory using the research dependency environment:

```bash
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python scripts/download_aps.py --help
PYTHONPATH=src python scripts/run_aps_extension.py --help
```

The original APS archive and raw CSVs are excluded. Download them separately and review configuration paths. This snapshot contains code and tests; it does not claim that a new APS experiment was run during upload.
