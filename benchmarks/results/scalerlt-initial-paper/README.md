# ScaleRLT initial-paper raw evidence

These immutable archives back the official Ouro inference baseline and repeated tiny learner control. They contain retained raw attempts and SHA manifests, with no checkpoint weight bodies. Historical device/source epochs stay distinct.

| Archive | SHA-256 | Scope |
| --- | --- | --- |
| official-c32-cost.tar.gz | 06d960ebb83e97eb1dee8af3a98b5c3cc956815eb558c7cdabccc37077ec5cb4 | 99 payloads: complete C32 fixed-work inference cost; reduced-depth quality remains uncertain |
| official-gsm8k-development.tar.gz | 2cd4fbfe157d3faa6889fb372f13ce3db6e0009de270ceec135b52f603effae2 | 106 payloads: all six arms × 128 frozen development IDs; strict independent grading |
| h20-current-b-control.tar.gz | d1dc4b22da2ce28d29e07106267ff5802134ab3faae4cade8d548dcec743e618 | 20 payloads: two tiny MCore CUDA controls/four paired AdamW updates |

Each archive has one additional hash manifest. Independent audit/handback records are adjacent. [Historical three-family CUDA data](../h20-mcore-all-families/README.md) and [CPU prefix controls](../packed-prefix-clean-cpu/summary-parent.json) are already retained. The [complete result table](../../../docs/scalerlt_initial_paper_results.md) states every speed denominator, quality status and pending endpoint. These archives do not establish full online RL or reward convergence.
