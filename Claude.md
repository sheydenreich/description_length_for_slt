Priorities for generated code:
- modular, easily expandable, minimal tech debt
- human understandable
- efficient computation

Rules for executing code:
- Anything heavy always via slurm jobs
- Nodes:
    - THIN (the node you are on) only for small CPU tests and smoke jobs, just 10 cores and limited RAM. Direct runs are allowed here.
    - EPYC for high-CPU tasks
    - GENOA for CPU tasks with high memory demand
    - GPU for small tests (V100)
    - DGX for regular GPU jobs (A100)
    - H100 if A100 is not sufficient (H100)
    - GPU allowance: up to 4 A100 simultaneously is always fine, 8 simultaneously for larger jobs, more only after asking
- Note that conda environments may not be compatible between nodes. Apptainer would be compatible, apptainer path is /u/lade/sheydenreich/fast/apptainer
