# Streamed readout: CPU forward/backward experiment

macOS arm64, Torch 2.13.0, FP32, one CPU thread; shared host, not reserved. Three independent process starts use seeds 31/32/33. Each shape/objective warms both arms, then runs three dense/streamed/streamed/dense blocks. Loss and both input/weight gradients are checked against dense computation in every observation. Times are geometric means over 18 observations per arm. The range contains the three process-level ratios, not a confidence interval.

| Response rows / vocabulary / width / tile | Objective | Dense fwd+bwd (ms) | Streamed fwd+bwd (ms) | Dense / streamed speed | Process ratio range | Saved intermediate bytes: dense → streamed |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 64 / 1024 / 128 / 256 | selected | 0.316 | 0.765 | 0.414× | [0.404, 0.421] | 262,144 → 1,024 |
| 64 / 1024 / 128 / 256 | selected + entropy + full KL | 0.656 | 1.478 | 0.444× | [0.436, 0.454] | 786,432 → 1,024 |
| 256 / 4096 / 256 / 512 | selected | 5.186 | 7.179 | 0.722× | [0.676, 0.763] | 4,194,304 → 4,096 |
| 256 / 4096 / 256 / 512 | selected + entropy + full KL | 10.284 | 15.720 | 0.654× | [0.641, 0.673] | 12,582,912 → 4,096 |

The Python vocabulary-tiled prototype is **slower in all four CPU cases**. Its saved intermediates are smaller. The byte count is the unique storage retained by autograd, excluding supplied inputs and parameters; it is not peak memory, CUDA memory or total training memory. Backward still allocates the parameter gradient and a vocabulary tile. These measurements exclude recurrent blocks, optimizer, sampling, rewards, publication and checkpoints. No GPU, model, full-RL or convergence improvement is established.

Reproduce with `PYTHONPATH=. OMP_NUM_THREADS=1 python benchmarks/bench_streamed_readout.py --seed SEED` for seeds 31, 32 and 33. Raw observations are committed in `benchmarks/results/streamed-readout-cpu/`.
