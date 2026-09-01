# FlexMaya RAS

This package contains the FlexEva RAS path used with the independently authored
Maya-style backend.

The V1 path is:

```text
fake-CUDA hooks -> C++ shared-memory event arena -> C++ trace RAS
    -> C++ rank dedup / compact trace -> one Python replay for prediction
```

Normal execution does not use JSONL.  JSONL-style trace files are only allowed
as explicit debug exports outside this package.

## Build And Test

```bash
python3 setup.py build_ext --inplace
python3 -m pytest -q
```

## GPT V1 Smoke

```bash
PYTHONPATH=src python3 examples/run_gpt_v1.py --synthetic
```

The real fake-CUDA route is wired through the compatibility C symbol
`plain_maya_hook_record_api_v2`, so the current fake-CUDA loader can use this
package's `_flexmaya_ras*.so` as its hook library while the public ABI remains
`flexmaya_record_api_v1`.
