# ScaleRLT — Current H20 native learner gate

The continued H20 (`GPU-8ee84e7d-143f-dd29-1097-85943783e027`) passes the original two CUDA/MCore/DDP/Adam configurations at source revision `d6229d638b08a3927d23d39854aa13a85df71933`. Each candidate performs two updates against its ordinary reference. The original full-gradient, parameter, Adam-moment, counter and logical-batch assertions remain unchanged.

| Reduction | Paired updates | Maximum gradient error | Maximum parameter error | Adam states and counters | Smallest update norm |
| --- | ---: | ---: | ---: | --- | ---: |
| Token mean | 2 | 7.451e-8 | 5.961e-8 | PASS | 0.027031 |
| Response mean | 2 | 1.192e-7 | 2.049e-8 | PASS | 0.028335 |

The supervisor completes naturally in 15.706 seconds. This is a verification duration, not a speed comparison. The complete 15,005-byte archive contains 20 payload files and one manifest, all independently hash verified. Launcher, supervisor, controller, test and finalizer are independently absent after completion; the original GPU and I/O locks are free and GPU compute is empty.

The first launch fails with SSH exit 255 after the own control connection expires. A new normal login confirms that all four own launch/output paths are absent before the next launch. Both attempts are retained. No process is killed and no environment is updated.

| Main native endpoint | State |
| --- | --- |
| Current-device tiny learner math and Adam gate | PASS |
| Pinned Thinking-model files, source and private disk preflight | PASS |
| Original continuous three updates, 96 training and 32 held-out samples | NOT_RUN |
| Fresh-worker recovery | NOT_RUN |
| Independent-seed reward convergence and simulated large-scale RL | NOT_RUN |

Evidence: `h20-epoch2-mcore-gate-whole-handback.json` and `h20-epoch2-mcore-gate-offbox/mcore-gate-epoch2-receipts.tar.gz`. Archive SHA-256: `d1dc4b22da2ce28d29e07106267ff5802134ab3faae4cade8d548dcec743e618`. These results are separate from the historical twelve-case run on the earlier physical GPU.
