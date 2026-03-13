# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-03-08

### Added
- `flash` attention type: Gated Attention Unit (GAU) from Hua et al., ICML 2022 — O(N·s) single-head gated linear attention with relu² feature map; controlled via `flash_query_key_dim` (default: `max(d_model // 2, 16)`)
- `performer` attention type: FAVOR+ from Choromanski et al., ICLR 2021 — O(N·r) unbiased softmax kernel approximation via orthogonal random features; controlled via `performer_num_random_features` (default: `max(head_dim * 4, 32)`)
- `bigbird` attention type: sparse BigBird attention (global + local-block + random) — O(N·√N) complexity
- `LinT`: single-modality transformer as a first-class model alongside `LinMulT`
- Auxiliary heads: per-branch output heads attached to the cross-modal stage via `auxiliary_heads` config
- Plain dicts accepted directly as head configs — `heads=[{"type": "simple", "output_dim": 5}]` works without wrapping in `HeadConfig`
- TAM validation at config construction: enabling `add_module_multimodal_signal` or `add_module_tam_fusion` without `tam_time_dim` now raises a clear `ValueError` immediately
- `py.typed` marker for PEP 561 type-checking support
- Sphinx + sphinx-autoapi + Furo documentation, deployed to GitHub Pages
- Jupyter notebook examples: `quick_start.ipynb`, `benchmark_urfunny.ipynb`, `benchmark_time_memory.ipynb`
- Python 3.13 support

### Changed
- Config key `dropout_relu` renamed to `dropout_ffn` (**breaking**)
- Config keys for attention tuning renamed for clarity (**breaking**): `query_key_dim` → `flash_query_key_dim`; `num_random_features` → `performer_num_random_features`
- `linmult/core/` restructured: `transformer.py` (encoder only), `temporal.py` (reducers + TRM + TAM), `norm.py` (BN/IN), `heads.py`; `modules.py` removed (**breaking** for direct sub-module imports)

### Removed
- `numpy` runtime dependency
- `dropout_residual` config key

---

## [1.8.0] — 2025-06-01

### Added
- Unimodal feature support: single-modality inputs handled natively
- Auxiliary heads: attach multiple output heads to a single backbone

---

## [1.7.0] — 2025-03-01

### Added
- Custom-named heads: specify `name` in head config instead of positional indexing

---

## [1.6.0] — 2024-12-01

### Added
- `time_dim_aligner` with `aap`, `amp`, and `padding` strategies for cross-modal temporal alignment
- `tam_fusion` option for temporal attention map-based fusion
- `multimodal_signal` flag for richer cross-modal interaction

---

## [1.5.2] — 2024-09-01

### Fixed
- NaN values during attention pooling in edge cases with very short sequences

---

[2.0.0]: https://github.com/fodorad/linmult/compare/v1.8.0...v2.0.0
[1.8.0]: https://github.com/fodorad/linmult/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/fodorad/linmult/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/fodorad/linmult/compare/v1.5.2...v1.6.0
[1.5.2]: https://github.com/fodorad/linmult/releases/tag/v1.5.2
