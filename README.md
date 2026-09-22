# DASE7506 MP1 - Zhang Yaxing (3036707458)

Frozen full-test CPU FP32 BPB: **1.5004815276**.
Submitted predictor: seed 17, step 18,800 EMA from a 24,000-step run.
Seed42 is supplementary validation-only evidence, not a replacement.

## Direct evaluation without retraining
Extract the code archive into a folder. Extract the checkpoint archive into that SAME
folder so it contains code/, REPORT.pdf, and checkpoints/final.pt.
All commands below run from code/. Use Python 3.12.

~~~sh
python -m venv .venv
~~~

Windows PowerShell activation: .venv\Scripts\Activate.ps1
Linux activation: source .venv/bin/activate

~~~sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python evaluate.py --checkpoint ../checkpoints/final.pt --device cpu --precision fp32 --threads 4 --split test --output ../peer-test.json
~~~

Expected BPB: 1.5004815276. Small hardware/build differences are possible.
Checkpoint SHA256: 770a45d67a74e16bffa065b00414be70654db64edda52ba6e44a2f7a6b8a46d5
Data/tokenizer/evaluator are included and unchanged; inference requires no network.
FILE_MANIFEST.json covers exact source/evidence bytes; ASSET_INVENTORY.json lists inference assets.

## Environment and retraining
Measured environment: Windows, Python 3.12.14, torch 2.7.1+cu128, numpy 2.5.3,
tokenizers 0.21.4; NVIDIA RTX 5080 training and Ryzen 7 9800X3D evaluation, four threads.
The existing environment was verified. A separate clean CPU-wheel install was not tested.

For the GPU build install torch with the cu128 index instead of the CPU index:
~~~sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python train.py --implementation student --config configs/final.json --device cuda --precision bf16 --threads 4 --seed 17 --steps 24000 --batch-size 32 --eval-every 400 --lr 0.003 --warmup 2400 --ema-decay 0.999 --run-dir runs/reproduce-final-s17
~~~

AdamW betas (0.9,0.95); wd=0.1 on >=2-D tensors only; norm scales exempt.
Gradient norm clip=1; EMA updated after optimizer step.
For zero-based step s:
lr(s)=0.003*min(1,(s+1)/2400)*(0.1+0.45*(1+cos(pi*s/24000))).
Warmup multiplies a cosine running from step zero.
best_checkpoint.pt is validation-selected EMA; checkpoint.pt is final-step EMA.
Full cost is 196,608,000 targets even though the best step occurs earlier.
No optimizer/RNG state is saved for exact interrupted resume.
CUDA retraining is not guaranteed bitwise identical. Change seed to 42 and use a new
run directory to reproduce the supplementary robustness experiment.

## Original baseline and equal-target control
The current train.py changed. Merely selecting --implementation model does NOT restore
the original baseline optimizer/schedule. README_STARTER.md is preserved historical guidance.
train_baseline_reference.py reconstructs the supplied old recipe: CPU FP32, 1200 steps,
batch32, seed17, AdamW default betas, lr1e-3, warmup100, all-parameter wd0.1, no EMA.
It is syntax-checked but was not fully retrained during packaging; original metrics remain evidence.
~~~sh
python train_baseline_reference.py --seed 17 --run-dir runs/reproduce-baseline
python train.py --implementation student --config configs/student.json --device cuda --precision bf16 --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 300 --lr 0.003 --warmup 120 --ema-decay 0.99 --run-dir runs/reproduce-eqtok
~~~
Both process 9,830,400 targets. Validation BPB: baseline 2.071084, early student 1.679617.
This comparison controls targets, not compute, architecture, precision or optimizer.

## Ablations and evidence
Use the final command with configs/abl_gelu_norope.json, configs/abl_gelu_noqk.json,
or configs/student_d8_drop015.json (SwiGLU), each in a new run directory.
REPORT.pdf explains outcomes and limitations. evidence/EXPERIMENT_LEDGER.json includes
all completed runs and corrects overconfident historical RUN_LOG narrative annotations.
Historical model source snapshots were not retained for every exploratory run.
implementation_sha256 identifies the model module, not the entire training program.
The local audit script expects original runs/ weights; peers need only evaluate.py.

## Resource verification
Three fresh-process full-test repeats per predictor, CPU FP32, four threads:
baseline median 6.201s, final median 18.807s,
ratio 3.033x (limit 5x). Final max peak working set 1.810 GiB
(limit 4 GiB), peak committed memory 2.881 GiB.
Memory uses Windows GetProcessMemoryInfo lifetime high-water marks including imports,
data/tokenizer loading and scoring. Timing is the evaluator's scoring timer only.
Conservative uncompressed assets including all benchmark data: 29.438 MiB (limit 64).
Repeated test scoring verifies the same frozen predictor; it does not select alternatives.

## AI assistance and reuse
Substantive ChatGPT/Codex assistance covered conceptual tutoring, architecture scaffolding
and code examples, review/debugging, training/evaluation execution, experiment analysis,
and report/documentation/packaging generation. The student implemented components during
guided work. The author must review and understand all submitted code and claims.
AI suggestions were tested: GELU replaced the initially recommended SwiGLU after experiments.
The supplied baseline/framework/data/tokenizer were reused. RMSNorm, RoPE and GLU
references are in REPORT.pdf. No pretrained weights or external training text were used.

## Data attribution
WikiText-2: Merity et al., https://arxiv.org/abs/1609.07843 .
Text by Wikipedia contributors; upstream https://huggingface.co/datasets/Salesforce/wikitext .
Retain CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/) and GNU FDL
(https://www.gnu.org/licenses/fdl-1.3.html) notices. Supplied revision:
b08601e04326c79dfdd32d625aee71d232d685c3.
These notices do not assign a new license to classroom code.
Original attribution and benchmark rules: code/README_STARTER.md and GUIDE.md.

## Publication
No public repository or release has been created yet.
Submit a code URL pinned to a full commit SHA and matching checkpoint release-asset URL.
See SUBMISSION_CHECKLIST.md. Review REPORT.pdf before publishing.
