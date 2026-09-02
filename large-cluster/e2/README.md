# E2 Figure 5 inputs

Figure 5(a)'s 32-, 64-, and 128-GPU points are replayed from the existing
real-GPU traces linked here as `gpt-32`, `gpt-64`, and `gpt-128`.

`figure5-source.json` records the paper-run predicted and measured runtimes and
their source SHA-256 identifiers. The default route derives the 32/64/128-GPU
actual runtimes from the trace step windows, checks them against those source
measurements, recomputes every error, and regenerates the CSVs and PDF. This is
a limited trace-based reproduction; generated result CSVs are not stored in
the repository.

The links fail when the supplied trace mount is absent. An equivalent mount
can be selected with `FIGURE5_LARGE_CLUSTER_ROOT`.

The independently trained estimator used by the evaluator is included as
`estimator.json` for the optional native route.
