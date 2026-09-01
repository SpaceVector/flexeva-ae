# E1 optimization history

E1 used `qwen/qwen3.5-plus-02-15` to generate four successive changes to the
same sparse-MoE workload. Each stage was then measured on the 128-GPU setup.
The prompts and parsed responses are stored beside this file; the executable
snapshots are under `script/e1/workload/`.

| Stage | Change | Step time | Estimated A2A bytes | Outcome |
| --- | --- | ---: | ---: | --- |
| Base | GShard-style top-2 routing with padded dispatch | 42.3347 s | 504,053,760 | Starting point |
| S1 | Packed variable-length dispatch | 30.1693 s | 41,634,588 | 28.74% faster than Base |
| S2 | Confident-primary fallback | 45.7717 s | 42,205,928 | Regression; retained as a failed iteration |
| S3 | Remove the fallback and use SDPA attention | 34.0680 s | 76,632,984 | Recovered from S2 |
| S4 | Switch-style top-1 routing | 16.3849 s | 25,050,836 | 61.30% faster and 95.03% less A2A than Base |

The first S4 response also changed `config.py`, which was outside the initial
editable-file list. The run was repeated after adding `config.py` and
`configs/gshard_style_fastloop.json` to that list. The retained S4 snapshot is
the accepted second response.
