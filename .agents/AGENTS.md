# Hemorrhage Research Framework Principles

These 15 principles govern the development and implementation of this research framework. They MUST be followed at all times.

1. **Root Cause Analysis (Mandatory):** Never implement a workaround. Always identify the root cause. Every modification must be justified.
2. **No Code Duplication:** If similar logic exists, refactor instead of copying.
3. **No Magic Numbers:** Every threshold, weight, hyperparameter, visualization frequency, or configuration must come from YAML configuration files.
4. **Production-Ready Modules:** Every new module must include type hints, docstrings, logging, error handling, unit-test compatibility, and configuration support.
5. **Configurable Research Features:** Every research feature (visualization, prediction reports, boundary analysis, wandb) must be optional and configurable. Training must still run without them.
6. **Backward Compatibility:** Every implementation must preserve backward compatibility. Existing checkpoints, configs, and experiments must continue to work whenever possible.
7. **Metric Validation:** Every new metric must be validated mathematically using synthetic tensors. Never trust a metric simply because the code runs.
8. **Publication-Quality Visualization:** Every visualization must be publication quality. No debugging plots. Generate figures suitable for thesis, journal, conference, and presentation.
9. **Verification Requirement:** Every implementation must include verification. No feature is considered complete until unit tests, integration tests, or controlled experiments prove correctness.
10. **Correctness Before Performance:** Performance comes after correctness. Do not optimize code until correctness has been verified.
11. **Evidence-Based Architecture:** Do not modify architecture unless supported by evidence. Architecture changes require separate experimental branches.
12. **Isolate Variables:** Never modify more than one scientific variable inside a single experiment. Each experiment should isolate the effect of one change whenever practical.
13. **Reproducibility Metadata:** Automatically generate reproducibility metadata. Every experiment must record git commit, git branch, configuration, dataset version, random seed, hardware, and software versions.
14. **Documentation as Implementation:** Documentation is part of implementation. Every major feature must include documentation, implementation notes, limitations, and future work.
15. **Future-Proof Scalability:** The framework should eventually support multiple datasets, multiple models, multiple losses, multiple metrics, and multiple visualization backends without requiring code changes.
