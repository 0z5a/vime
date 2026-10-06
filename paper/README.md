# ScaleRLT initial research manuscript

**ScaleRLT: I/O-Aware End-to-End Reinforcement Learning for Recurrent Looped Transformers**

Author: 0z5a. Draft v0.1, October 6, 2026.

[Read the eight-page PDF](ScaleRLT_v0.1_2026-10-06.pdf), [edit the standalone LaTeX](scalerlt.tex), or [inspect the Markdown result tables](../docs/scalerlt_initial_paper_results.md).

The draft contains twelve tables, a vector system diagram, seven displayed equations and a reproducibility appendix. It preserves CPU negative controls, official inference quality uncertainty, BF16 failures and strict IQuest update failures. Complete online RL, fresh-worker recovery, high-load/multi-GPU scaling and reward convergence remain pending. The [execution priority](../docs/scalerlt_priority_execution.md) is RFC465 standard backend acceptance, actual reward-correct scaling and a trained loop-native compute policy using merged PR454.

The standalone source compiles with the native LaTeX editor. The portable PDF uses the same compiled source with ReportLab two-column typesetting and STIX math rendering; the native compiler returns diagnostics without PDF export. Existing runtimes are used, without installing or updating an environment. All eight pages are rasterized and visually inspected; title/author, numeric/scope text, hash binding, text geometry and orphan pages are checked. [Compile/export receipts](../artifacts/scalerlt-initial-paper-v01) record the exact source/PDF identities.

With an existing Python containing Matplotlib, Pillow, ReportLab and pypdf, reproduce the portable export from the repository root:

```bash
python paper/export_initial_pdf.py paper/scalerlt.tex ScaleRLT-v01.pdf artifacts/scalerlt-initial-paper-v01/generation.json
```

The LaTeX source is the authoritative editable artifact. Numerical evidence and matched speed tables are linked from the result ledger; no component ratio is promoted to complete-RL acceleration.
