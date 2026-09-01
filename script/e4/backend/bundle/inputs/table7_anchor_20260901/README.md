# Table 7 ASTRA-Sim anchors

These replay caches are the baseline RAS states for Table 7. The driver
regenerates the baseline and candidate Chakra ET inputs, verifies their
fingerprints, builds each reuse plan, and runs full and RAS simulations against
the same candidate prefix.

The GPT cache uses the paper GPT-3 2.7B configuration: 16 modeled ranks,
TP1/PP8/DP2, global batch size 512, and 256 micro-batches. It contains 80,418
completed partitions, including 16 data-parallel collective partitions, with
no missing finish events.

The Routed-MoE cache is the retained baseline used by the six backend cases.
`manifest.json` records both cache hashes, context fingerprints, partition
counts, and regenerated-baseline fingerprints. The caches are experiment
inputs, not reported Table 7 measurements.
