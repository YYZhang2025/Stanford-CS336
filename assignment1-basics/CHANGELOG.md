# Changelog

All changes we make to the assignment code or PDF will be documented in this file.

## [1.0.6] 2025-08-28
- handout: Fix bug in RoPE formulation
- code: Fix typing for adapters
- code: Relock dependencies
- code: Minor reformatting


## [1.0.5] 2025-04-15
- code: Add submission script, fix typos
- code: Use `uv_build` for the package build system
- handout: Fix RoPE indexing
- code: Fix test for truncated LM inputs
- code: Ruff reformatting and linting
- code: Simplify snapshot testing
- code: Make everything compatible with `ty` typing

## [1.0.4] - 2025-04-08
### Added
- handout: add guidance on parallelizing pretokenization and provide starter code for chunking
- handout: add guidance on removing special tokens before pretokenization (you should split on them!)

### Changed
- handout: fix command for compiling model on MPS backend

## [1.0.3] - 2025-04-07
### Added
- code: Test for removing special tokens when training BPE

### Changed
- handout: Fixed RoPE off-by-one error
- code: Fix for Intel Macs, support 3.11 and PyTorch 2.2.2 when on MacOS x86_64

## [1.0.2] - 2025-04-03
### Added
- code: Missing tests for Linear and Embedding

### Changed
- handout: Fix RMSNorm interface
- handout: Add hints in RMSNorm and SwiGLU specifications to improve numerical stability
- handout: Clarify the BPE stylized example uses naive pretokenization by splitting on whitespace; your implementations should still use the provided regex
- handout: Fix docstring for RMSNorm

## [1.0.1] - 2025-04-02
### Changed
- code: Add some tolerance to RoPE tests
- code: Make sure loading happens on CPU for tests
- code: Fix docstring typing for readability of Tensor annotations in VSCode (pls fix, @Microsoft)

## [1.0.0] - 2025-04-01

### Added
- code: uv environments
- code: Tensor type annotations
- code: Snapshot testing, use --update-snapshots to update (soln only)
- code/handout: SwiGLU
- code/handout: RoPE
- handout: einops
- code/handout: new ablations
- handout: low-resource tips
- handout: Linear and Embedding from scratch

### Changed
- handout: complete reformatting, restructuring, rewording
- handout: redefine attention
- handout: GeLU -> SiLU
- handout: parameter initialization
- handout: remove dropout
- handout: redraw figures
- handout: new experiments
- handout: BPE explanations, systems, explain weird behaviors in solutions

### Fixed

- code: fix `test_get_batch` to handle "AssertionError: Torch not compiled with CUDA enabled".
- handout: clarify that gradient clipping norm is calculated over all the parameters.
- code: fix gradient clipping test comparing wrong tensors
- code: test skipping parameters with no gradient and properly computing norm with multiple parameters

## [0.1.6] - 2024-04-13

### Added

### Changed

### Fixed

- handout: edit expected TinyStories run time to 30-40 minutes.
- handout: add more details about how to use `np.memmap` or the `mmap_mode` flag
  to `np.load`.
- code: fix `get_tokenizer()` docstring.
- handout: specify that problem `main_experiment` should use the same settings
  as TinyStories.
- code: replace mentions of layernorm with RMSNorm.

## [0.1.5] - 2024-04-06

### Added

### Changed

- handout: clarify example of preferring lexicographically greater merges to
  specify that we want tuple comparison.

### Fixed

- handout: fix expected number of training tokens for TinyStories, should be
  327,680,000.
- code: fix typo in `run_get_lr_cosine_schedule` return docstring.
- code: fix typo in `test_tokenizer.py`

## [0.1.4] - 2024-04-04

### Added

### Changed

- code: skip `Tokenizer` memory-related tests on non-Linux systems, since
  support for RLIMIT_AS is inconsistent.
- code: reduce increase atol on end-to-end Transformer forward pass tests.
- code: remove dropout in model-related tests to improve determinism across
  platforms.
- code: add `attn_pdrop` to `run_multihead_self_attention` adapter.
- code: clarify `{q,k,v}_proj` dimension orders in the adapters.
- code: increase atol on cross-entropy tests
- code: remove unnecessary warning in `test_get_lr_cosine_schedule`

### Fixed

- handout: fix signature of `Tokenizer.__init__` to include `self`.
- handout: mention that `Tokenizer.from_files` should be a class method.
- handout: clarify list of model hyperparameters listed in `adamwAccounting`.
- handout: clarify that `adamwAccounting` (b) considers a GPT-2 XL-shaped model
  (with our architecture), not necessarily the literal GPT-2 XL model.
- handout: moved softmax problem to where softmax is first mentioned (Scaled Dot-Product Attention, Section 3.4.3)
- handout: removed redundant initialization (t = 0) in AdamW pseudocode
- handout: added resources needed for BPE training

## [0.1.3] - 2024-04-02

### Added

### Changed

- handout: edit `adamWAccounting`, part (d) to define MFU and mention that the
  backward pass is typically assumed to have twice the FLOPS of the forward pass.
- handout: provide a hint about desired behavior when a user passes in input IDs
  to `Tokenizer.decode` that correspond to invalid UTF-8 bytes.

### Fixed

## [0.1.2] - 2024-04-02

### Added

- handout: added some more information about submitting to the leaderboard.

### Changed

### Fixed

## [0.1.1] - 2024-04-01

### Added

- code: add a note to README.md that pull requests and issues are welcome and
  encouraged.

### Changed

- handout: edit motivation for pre-tokenization to include a note about
  desired behavior with tokens that differ only in punctuation.
- handout: remove total number of points after each section.
- handout: mention that large language models (e.g., LLaMA and GPT-3) often use
  AdamW betas of (0.9, 0.95) (in contrast to the PyTorch defaults of (0.9, 0.999)).
- handout: explicitly mention the deliverable in the `adamw` problem.
- code: rename `test_serialization::test_checkpoint` to
  `test_serialization::test_checkpointing` to match the handout.
- code: slightly relax the time limit in `test_train_bpe_speed`.

### Fixed

- code: fix an issue in the `train_bpe` tests where the expected merges and vocab did
  not properly reflect tiebreaking with the lexicographically greatest pair.
  - This occurred because our reference implementation (which checks against HF)
    follows the GPT-2 tokenizer in remapping bytes that aren't human-readable to
    printable unicode strings. To match the HF code, we were erroneously tiebreaking
    on this remapped unicode representation instead of the original bytes.
- handout: fix the expected number of non-embedding parameters for model with
  recommended TinyStories hyperparameters (section 7.2).
- handout: replace `<|endofsequence|>` with `<|endoftext|>` in the `decoding` problem.
- code: fix the setup command (`pip install -e .'[test]'`)to improve zsh compatibility.
- handout: fix various trivial typos and formatting errors.

## [0.1.0] - 2024-04-01

Initial release.
