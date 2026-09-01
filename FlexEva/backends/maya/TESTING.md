# Testing

Run portable checks from the core repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
make PYTHON=.venv/bin/python test
```

The first test is CPU-only and should report one passing test plus
`ras-slots=code,semantic,runtime-values,trace`. Full evaluator experiments and
their CUDA/NCCL environment policy belong to the consuming project.
