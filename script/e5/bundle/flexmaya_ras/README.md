# FlexMaya RAS (frozen E5 source)

This package is the evaluator implementation retained for the submitted-value
Table 8 audit. The primary artifact implementation is under `FlexEva/`; this
copy remains frozen so the archived measurement can be reproduced against its
original source.

The data path is:

```text
fake-CUDA hooks -> shared-memory event arena -> trace RAS
    -> rank deduplication -> compact trace -> Python replay
```

Build and test from this directory:

```bash
python3 setup.py build_ext --inplace
python3 -m pytest -q
```
