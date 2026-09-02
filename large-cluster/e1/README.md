# E1 supplied raw traces

The reviewer server has 16 GPUs, so E1's five 128-GPU rounds cannot be
recaptured there. `historical_sparse_moe` links to the raw trace mount provided
on the evaluation server. `script/run_e1` validates all ranks and measured-step
windows before extracting timing and communication data.

Set `E1_TRACE_ROOT` when the same trace tree is mounted elsewhere.
