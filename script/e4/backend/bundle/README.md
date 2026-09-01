# Table 7 ASTRA-Sim source bundle

This directory contains the pinned inputs needed to build and run Table 7:

- ASTRA-Sim commit `518bd513ae110428cd62eb60efc0f3993fd53c70`;
- Chakra commit `0e3cd40c569f0a4cacb6d961bb56be53407abd2f`;
- the RAS patch and source overlay;
- GPT and Routed-MoE replay-cache inputs; and
- yaml-cpp commit `a83cd31548b19d50f3f983b069dceb4f4d50756d`
  under `vendor/` for an offline dependency build.

From the artifact root:

```bash
script/e4/backend/setup_astra.sh build
script/e4/backend/setup_astra.sh check
```

The setup script clones the pinned ASTRA-Sim source below `.deps/`, verifies
the bundled hashes, applies the patch and overlay, builds the ns-3 backend, and
runs the driver self-test. `script/run_e4` then runs all six Table 7 cases and
writes results below `result/e4/generated/table7/`.
