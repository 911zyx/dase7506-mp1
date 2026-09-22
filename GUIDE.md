# MP1 — Small Language Model Challenge

DASE7506 · Individual coursework  
**Final submission deadline: end of 30 September 2026 (UTC+8).**

Train a language model from scratch, improve its architecture or training, and achieve the lowest reproducible test bits per byte within the resource limits.

## 1. What we provide

- **guide:** this assignment guide.
- **code:** a complete baseline GPT, training and evaluation code, data, tokenizer and correctness tests. The [code README](../code/README.md) explains installation, commands and measurement procedures. No pretrained checkpoint is supplied; training starts from random initialization.

## 2. Ranking and assessment

- **Leaderboard:** full-test bits per byte (BPB), **lower is better**, on the supplied WikiText-2 benchmark with its fixed BPE-2048 tokenizer. Use FP32 evaluation and report a score reproducible on CPU. The baseline achieves approximately **2.10 BPB**.
- **Experimental evidence:** report the initial baseline, a comparison using the same number of processed training targets, and an ablation of your key mechanism. One paired experiment may satisfy both controls. Use validation for development and selection.  Explain why the method helps or fails, what the comparisons establish, and the trade-offs between prediction quality and computational cost. Rankings remain provisional until verification and public review finish.

 Grade conversion and bonus amounts will be announced separately.

## 3. Constraints

- **Improve the model and training:** the test-time budget keeps evaluation lightweight and discourages reliance on large retrieval databases or lookup indexes.
- **Evaluation budget:** at most **5× baseline CPU scoring time, 4 GiB peak evaluation RAM and 64 MiB of uncompressed inference assets**. Follow the measurement procedure in the code README.
- **Data and causality:** learn only from the supplied training text; select settings on validation. Keep the data, tokenizer and evaluator unchanged. No external training text, pretrained weights, test-based tuning, cached validation/test answers, future-token access, cross-window state or evaluation network access. Evaluation uses independent causal windows of 256 tokens. Freeze the method before testing.
- **Training:** CPU or GPU; course GPUs are not provided. Training duration and architectures are unrestricted within the evaluation limits. Disclose training and search costs.
- **AI assistance is allowed:** understand your implementation, acknowledge reused work and disclose substantive AI help in your repository README.

## 4. Submission and peer review

- **Before 29 September:** submit your **student ID and full-test BPB** on the [course website](https://xudongwu-0.github.io/courses/dase7506/); links are optional. Finish creating the generated GitHub issue. Use the same GitHub account for updates.
- **Final submission, by 30 September:** your **last submission must include both links**: immutable code with exact installation/training/evaluation instructions, and the matching checkpoint bundle for evaluation **without retraining**. Include a report of **at most 10 pages** in the code repository, covering method, comparisons, ablation and critical analysis. Scores and model bundles freeze at the deadline.
- **Public review:** after **30 September**, all links and frozen scores are released together for **7 days**. Links submitted through the form remain encrypted until release.
- **Peer review and rewards:** evaluate another student's checkpoint with the supplied full-test FP32 scorer. Click **Peer Review Report** beside their result; the **reproduced score is required**, while Markdown details and evidence links are optional. Alternatively, email [wu.xudong@connect.hku.hk](mailto:wu.xudong@connect.hku.hk) with the submission number and reproduced score. Instructor-confirmed discrepancy reports during public review earn **bonus credit**. Small numerical differences or installation failures alone do not establish misconduct.
🔴 We use automated checkpoint evaluation scripts to verify scores, alongside instructor spot checks. Confirmed serious score misreporting will be formally recorded.
