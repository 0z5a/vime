# ScaleRLT — H20 MCore two-update correctness

All twelve frozen tiny FP32 CUDA configurations passed on the original H20 using the existing 0z5a environment, source `2ad271d5fbb58f06170a05277b03663bf9d66b24`. Each candidate performs two actual MCore/DDP/AdamW updates against the ordinary replay oracle. Both token-mean and response-mean losses are included. The original full-gradient, parameter, Adam first/second-moment, counter and sample-accounting assertions are unchanged.

| Family / schedule | Candidate updates | Maximum main-gradient error | Maximum parameter error | Complete update gate | Speed ratio |
| --- | ---: | ---: | ---: | --- | --- |
| Ouro ordinary | 4 | 1.192093e-07 | 5.960464e-08 | PASS | NOT_MEASURED |
| Ouro joint | 4 | 1.192093e-07 | 5.960464e-08 | PASS | NOT_MEASURED |
| Nanbeige ordinary | 4 | 5.960464e-08 | 2.980232e-08 | PASS | NOT_MEASURED |
| Nanbeige joint | 4 | 5.960464e-08 | 2.980232e-08 | PASS | NOT_MEASURED |
| Huginn independent latent | 4 | 7.152557e-07 | 9.302312e-08 | PASS | NOT_MEASURED |
| Huginn mixed latent | 4 | 1.177192e-06 | 2.094894e-07 | PASS | NOT_MEASURED |

Pytest reports **12 passed, 36 warnings in 22.66s**. This is verification duration, not a baseline/candidate speed comparison. Twenty-four candidate updates and their matched ordinary references use tiny fixed trajectories; they do not establish official checkpoint accuracy, online reward convergence, memory saving or large-scale stability. Existing CPU BF16 failures remain failed.

The natural controller, test, wrapper and finalizer exits are recorded; all four recorded PIDs are absent in a fresh read-only check and the compute list is empty. The offbox archive is 21,546 bytes with 84 verified payloads plus its manifest, SHA-256 `5cbea08375482e57936c9808a352fa3d12b22aabbb4a04a21723deac117f42ff`. Raw data: [`cuda-gate.tar.gz`](cuda-gate.tar.gz); independent handback: [`independent-handback.json`](independent-handback.json).
