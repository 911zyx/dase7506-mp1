# Small language modeling under a CPU budget

Zhang Yaxing | 3036707458

DASE7506 MP1 | Zhang Yaxing | 3036707458<br/>Frozen submission and reproducibility report | 22 September 2026

## 1. Result and task

The frozen 3.94M-parameter predictor achieves 1.500482 full-test bits per byte (BPB), versus 2.101260 for the supplied baseline: a 28.59% reduction. This combines architecture, regularization and longer training; it is not an architecture-only gain.

Training uses only supplied WikiText-2 [1] with fixed BPE-2048. All model selection uses validation. Independent causal windows have 256 next-token targets; every target except the first split token is scored, including the last partial window. BPB divides summed negative log-base-2 probability by the entire split byte count. Ranked inference uses CPU FP32.

| Split | Targets | UTF-8 bytes |
| --- | --- | --- |
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

## 2. Architecture

Eight pre-normalized Transformer blocks use width 192, six heads (32 channels/head) and 3,935,936 unique parameters. Embedding and output weights are tied; linear biases are absent. Dropout 0.15 applies to attention probabilities and both residual outputs during training only.

RMSNorm [2] divides each token vector by its root mean square with epsilon 1e-6 and learned channel scales; it does not subtract the mean. Queries and keys additionally receive per-head RMSNorm before RoPE [3]. Rotary angles use position times 10000^(-2j/32) with first/second-half channel pairs. GELU hidden width 768 is parameter-matched to the explored SwiGLU-512 [4].

Weights start from N(0,0.02); residual output matrices are reinitialized at 0.02/sqrt(16)=0.005. This reduces initial branch scale, not a guarantee of bounded activations or stability. The contribution is the evaluated combination under the resource budget, not invention of these components.


---

## 3. Training and experimental design

AdamW uses betas (0.9,0.95), weight decay 0.1 on matrices only, and gradient-norm clipping at 1. BF16 autocast trains on an RTX 5080; validation uses FP32. Seed17, batch32 and 24,000 updates process 196,608,000 targets. Windows are sampled with replacement using a generator separate from model initialization.

At zero-based step s, lr=0.003 x min(1,(s+1)/2400) x [0.1+0.45(1+cos(pi s/24000))]. Warmup multiplies an already-running cosine. EMA starts as a model copy, updated after each optimizer step: ema=0.999 ema+0.001 weights. Validation runs every 400 updates; the lowest-BPB EMA snapshot is retained.

The submitted run selected step18,800 (154,009,600 targets), but its full 24,000-step cost is reported. Test scoring followed checkpoint selection. Later repeated tests verify this identical frozen predictor only. Seed42 was a later fixed-recipe, validation-only robustness check.

## 3.1 Equal processed-target control

| Run | Targets | Parameters | Val BPB |
| --- | --- | --- | --- |
| Supplied baseline | 9,830,400 | 1,088,256 | 2.071084 |
| Early 6-layer SwiGLU student | 9,830,400 | 3,050,304 | 1.679617 |
| Final 8-layer GELU, seed17 | 196,608,000 | 3,935,936 | 1.482680 |

The equal-target control shows an improvement at equal target count, while architecture, capacity, optimizer, schedule, averaging and precision all change. It does not control FLOPs or isolate one mechanism. The final run processes 20 times the baseline targets.

## 3.2 Reproduction limits

Frozen-weight CPU rescoring is more reproducible than GPU retraining. A fixed CUDA seed does not guarantee identical trajectories. Historical source snapshots were not retained for every exploration, and model-module hashes do not fingerprint the full trainer. Current configs, exact final commands and original metrics are supplied; byte-identical historical retraining is not claimed.


---

## 4. Mechanism ablations

These runs share seed17, eight layers, dropout0.15, 24,000 updates, schedule and EMA0.999. Each reports its own validation-selected snapshot, so this compares matched training procedures, not identical checkpoint steps. Most pairs have only one seed.

| Variant | Val BPB | Delta | Parameters | Train s |
| --- | --- | --- | --- | --- |
| GELU base | 1.481952 | +0.000000 | 3,935,936 | 505.5 |
| SwiGLU, matched MLP | 1.493970 | +0.012018 | 3,935,936 | 544.5 |
| Learned absolute positions | 1.486995 | +0.005043 | 3,985,088 | 401.2 |
| Without QK normalization | 1.492693 | +0.010741 | 3,935,424 | 349.0 |

GELU beat SwiGLU by about 0.012 BPB, reversing the initial expectation. Published GLU benefits [4] do not guarantee a gain here. Learned absolute positions were about 0.005 BPB worse and added 49,152 parameters, so that positional ablation is not exactly parameter-matched.

Removing QK normalization worsened validation by about 0.011 without observed training failure. Its evidence here is quality, not necessity to prevent divergence. Simpler variants trained faster, but single-run times mix implementation costs and system variation.

## 4.1 EMA at the same final update

| GELU run, step24,000 | Val BPB |
| --- | --- |
| Raw weights | 1.498944 |
| EMA weights | 1.482831 |
| Raw minus EMA | 0.016112 |

This pair isolates using averaged weights within one trajectory at the same update. Comparing final raw weights against the best earlier EMA would mix averaging with checkpoint selection.

RMSNorm, weight tying and residual initialization were not individually ablated. No isolated benefit is claimed for them. Single-seed differences are observations, not significance tests.


---

## 5. Validation trajectories and robustness

Figure: validation curves from final-d8gelu-24k-s17 and s42 metrics, updates >=4000.

| Frozen recipe | Best val BPB | Selected step |
| --- | --- | --- |
| Seed17 (submitted) | 1.482680 | 18,800 |
| Seed42 (supplementary) | 1.484693 | 19,200 |

The difference is 0.002013 BPB (about 0.14%). Similar minima and selected steps provide limited two-run robustness evidence. Two seeds do not establish a general noise floor or significance for all ablations. Seed42 was run after test disclosure and does not replace the frozen submission.

## 5.1 Exploration and limitations

Without dropout, longer training exposed overfitting. Dropout delayed deterioration and longer schedules improved validation. Changing total schedule length also changes learning rates earlier in the run: a 24k schedule is not simply a continuation of an 8k schedule. Depth/dropout/schedule probes are exploratory, not a full factorial design.

Selection among many validation checkpoints introduces optimism. Most ablations use one seed. Timing measurements are local to this hardware. The experiment ledger preserves all completed runs and corrects overconfident interpretations in historical notes.


---

## 6. Resource verification and integrity

The submitted checkpoint remains byte-identical. Protected source, evaluator, baseline model/config, tests, requirements and all data match the supplied manifest. Five default and five actual-final-config tests passed: causality, probability normalization, sample/window independence, gradients and target counting.

| Measurement | Observed | Limit |
| --- | --- | --- |
| Baseline CPU median, 3 repeats | 6.201 s | Reference |
| Final CPU median, 3 repeats | 18.807 s | 5x baseline |
| Matched time ratio | 3.033x | <=5x |
| Maximum peak working set | 1.810 GiB | <=4 GiB |
| Checkpoint file | 16.535 MiB | Part of assets |
| Conservative inference assets | 29.438 MiB | <=64 MiB |

Fresh-process CPU FP32 scoring uses four threads on the Ryzen 7 9800X3D. Time is the fixed scorer timer, excluding loading. Windows PeakWorkingSetSize measures process-lifetime peak physical memory, including imports, loading and tokenization; peak committed memory was 2.881 GiB. These are not GPU allocation measurements.

Three frozen final scores match the original test BPB within 1e-7. Asset accounting conservatively includes all benchmark splits, tokenizer, inference source and weights, excluding installed runtimes and documentation. Timing is hardware-dependent; matching CPU settings matters.

## 6.1 Search costs

There are 26 completed local runs with metrics (23 non-smoke, including baseline and later seed42). Total processed targets: 2,877,276,160. Summed GPU training: 105.53 minutes; GPU process time: 115.00 minutes. CPU training adds 4.96 minutes.

Totals include complete runs even after the selected checkpoint, and smoke checks. They exclude installation, human/AI time, undocumented failed attempts and separate scoring repeats. This is not total project wall-clock time. All runs started from random initialization; no pretrained or external-text ancestry is used.


---

## 7. Reproduction and release identity

Extract both archives into one root and use the unchanged evaluator from code/ against ../checkpoints/final.pt. README.md contains installation, training, baseline reconstruction and ablation commands. File manifests and archive SHA256 values identify exact bytes.

Submitted checkpoint SHA256<br/><font size="8">770a45d67a74e16bffa065b00414be70<br/>654db64edda52ba6e44a2f7a6b8a46d5</font>

The final config keeps mlp_hidden=512; mlp="gelu" expands it internally to 768 to match SwiGLU parameters. Exploratory weights are omitted from the release; recorded metrics are included. The baseline helper reconstructs the original recipe and does not replace original measured evidence.

## 8. AI assistance and review

Substantive ChatGPT/Codex assistance included tutoring, architecture scaffolding and examples, debugging/review, running training/evaluation, interpreting results, and generating report/documentation/packaging. The student implemented components during guided work. The author must review and understand the final submission; this disclosure is also in README.md.

The principal contribution is an evaluated adaptation of established components to a small-data, CPU-budget setting, including a negative SwiGLU result and explicit overfitting/EMA analysis.

## References

[1] Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher. Pointer Sentinel Mixture Models. 2016. https://arxiv.org/abs/1609.07843

[2] Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. NeurIPS 2019. https://arxiv.org/abs/1910.07467

[3] Jianlin Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. 2021. https://arxiv.org/abs/2104.09864

[4] Noam Shazeer. GLU Variants Improve Transformer. 2020. https://arxiv.org/abs/2002.05202

[5] Supplied DASE7506 MP1 GUIDE.md, starter README, evaluator and package manifest. Included course materials.

[6] evidence/EXPERIMENT_LEDGER.json, per-run metrics and resource_audit/audit_summary.json.
