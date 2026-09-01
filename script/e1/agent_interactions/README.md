# E1 agent interaction records

This directory preserves the original OpenRouter/Qwen exchanges that produced
the four E1 optimization stages. They are provenance records, not an input to
`script/run_e1`; trace and real modes execute the retained workload snapshots
under `script/e1/workload/`.

| E1 stage | Record directory |
| --- | --- |
| S1 | `round_01_packed_variable_dispatch/` |
| S2 | `round_02_top1_confident_fallback/` |
| S3 | `round_03_sdpa_attention_recovery/` |
| S4 | `round_04_switch_top1_routing/` |

Each round contains the system prompt, human optimization brief, fully
assembled user prompt, raw OpenRouter response, and parsed agent response.
`PROCESS_LOG.md` records the chronological loop. Base has no interaction files
because it is the pre-agent anchor.

The records are preserved from the original experiment workspace. The first
S4 response required a wider editable-file list; `PROCESS_LOG.md` summarizes
the accepted rerun. The rejected response was not retained separately.
