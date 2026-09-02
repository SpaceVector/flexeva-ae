# FlexMaya RAS (E5 source snapshot)

This is the FlexMaya RAS source used by E5. The main implementation is under
`FlexEva/`; E5 stages this snapshot so every run uses the same source.

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
