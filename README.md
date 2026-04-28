# EML Circuit

Residual EML circuit layers, an EML tree-search baseline, and benchmark runners for comparing against MLPs.

연구 배경과 설계 메모는 [docs/research_notes.md](/mnt/d/GitHub/Eml_Circuit/docs/research_notes.md)에 정리했습니다. 이 README는 설치, 실행, 벤치마크 사용법 중심입니다.

## Install

요구사항:

- Python `>=3.10`
- `torch>=2.2`
- `tqdm>=4.66`

개발 환경 예시:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## What Is Included

- `emlstack`: residual EML layer 기반 회귀 모델
- `mlp`: GELU 기반 MLP baseline
- `eml_tree`: EML tree beam-search baseline
- synthetic benchmark suites for `tree` and `mlp`

현재 기본 실험 크기:

- `hidden_dim=16`
- `depth=2`
- `width=None`
- `normalize_targets=auto`

`emlstack`에서 `width=None`이면 내부 EML 폭은 `max(4, hidden_dim // 2)`로 자동 결정됩니다.
`normalize_targets=auto`이면 `tree` benchmark는 train target 기준 z-score 정규화를 적용하고, `mlp` benchmark는 원래 스케일을 유지합니다.

핵심 모듈:

- [eml_circuit/layers.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/layers.py)
- [eml_circuit/models.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/models.py)
- [eml_circuit/symbolic.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/symbolic.py)
- [eml_circuit/benchmarks.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/benchmarks.py)
- [eml_circuit/training.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/training.py)

## Benchmarks

벤치마크는 2개 그룹으로 나뉩니다.

### `tree` group

`eml_tree`와 `emlstack` 비교용입니다. EML 구조를 직접 재사용하는 함수들입니다.

- `tree_shared_subexpr_a`
- `tree_deep_chain_b`
- `tree_circuit_reuse_c`

### `mlp` group

`mlp`와 `emlstack` 비교용입니다. 타깃 함수에 `GELU`, `ReLU`, `SiLU`를 직접 사용합니다.

- `mlp_gelu_mix_a`
- `mlp_relu_piecewise_b`
- `mlp_silu_gate_c`

벤치마크 정의는 [eml_circuit/benchmarks.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/benchmarks.py:34)에 있습니다.

이전 이름도 일부 유지합니다.

- `shared` -> `tree_shared_subexpr_a`
- `deep` -> `tree_deep_chain_b`
- `circuit` -> `tree_circuit_reuse_c`

## Quick Start

단일 실험:

```bash
python3 examples/run_emlstack_demo.py \
  --benchmark tree_shared_subexpr_a \
  --model emlstack
```

MLP benchmark에서 `mlp` baseline 실행:

```bash
python3 examples/run_emlstack_demo.py \
  --benchmark mlp_relu_piecewise_b \
  --model mlp
```

EML tree baseline 실행:

```bash
python3 examples/run_emlstack_demo.py \
  --benchmark tree_circuit_reuse_c \
  --model eml_tree
```

## Benchmark Suite

그룹 전체 실행:

```bash
python3 examples/run_benchmark_suite.py --benchmark-group tree
python3 examples/run_benchmark_suite.py --benchmark-group mlp
```

특정 benchmark들과 모델 조합만 실행:

```bash
python3 examples/run_benchmark_suite.py \
  --benchmarks tree_shared_subexpr_a mlp_gelu_mix_a \
  --models emlstack mlp eml_tree
```

GPU가 여러 개 있으면 가능한 만큼 프로세스를 분리해서 각 작업을 디바이스에 배치합니다. GPU가 2개이고 작업이 여러 개면 2개가 병렬로 돌고, 남은 작업은 슬롯이 비면 이어서 실행합니다.

병렬 실행 시 epoch 로그는 작업별 파일로 분리됩니다.

- `--save-dir`를 주면 로그는 `SAVE_DIR/logs/*.log`
- `--save-dir`가 없으면 기본 `suite_logs/*.log`
- 직접 지정하려면 `--log-dir`

## Useful Options

공통 옵션:

- `--n-train`
- `--n-extrap`
- `--hidden-dim`
- `--depth`
- `--width`
- `--epochs`
- `--batch-size`
- `--lr`
- `--device`
- `--save-path` or `--save-dir`
- `--log-dir`
- `--normalize-targets`
- `--no-progress`

EML tree baseline 전용 옵션:

- `--tree-max-depth`
- `--tree-beam-width`
- `--tree-max-basis-size`
- `--tree-min-improvement`
- `--tree-selection-pool-size`

suite sweep 옵션:

- `--hidden-dims`
- `--widths`

EMLStack / MLP 학습 로그는 `tqdm` 진행바와 epoch metric으로 출력됩니다. 체크포인트에는 모델 상태와 metric history가 함께 저장됩니다.

## Example Commands

EMLStack vs EML tree:

```bash
python3 examples/run_benchmark_suite.py \
  --benchmark-group tree \
  --models emlstack eml_tree \
  --save-dir checkpoints/tree_suite
```

EMLStack vs MLP:

```bash
python3 examples/run_benchmark_suite.py \
  --benchmark-group mlp \
  --models emlstack mlp \
  --save-dir checkpoints/mlp_suite
```

2개 GPU를 명시적으로 지정:

```bash
python3 examples/run_benchmark_suite.py \
  --benchmark-group mlp \
  --devices cuda:0 cuda:1
```

여러 capacity sweep 실행:

```bash
python3 examples/run_benchmark_suite.py \
  --benchmark-group tree \
  --models emlstack eml_tree \
  --hidden-dims 8 16 \
  --widths 4 8
```

더 작은 EMLStack으로 실행:

```bash
python3 examples/run_emlstack_demo.py \
  --benchmark tree_deep_chain_b \
  --model emlstack \
  --hidden-dim 8 \
  --depth 2 \
  --width 4
```

## Outputs

단일 실행 스크립트는 다음을 출력합니다.

- `benchmark`
- `model`
- `device`
- `hidden_dim`
- `width`
- `trainable_parameters`
- `train_mse`
- `extrap_mse`
- `best_epoch`
- `best_score`

suite 실행 summary도 동일하게 `hidden_dim`, `width`, `params`를 함께 출력합니다.

병렬 suite 실행 시 추가 출력:

- `log`

`eml_tree` 실행 시 추가 출력:

- `selected_expression_count`
- `selected_total_nodes`
- `selected_max_depth`

체크포인트 저장 시 포함되는 내용:

- `model_state_dict`
- `config`
- `metrics`
- `trainable_parameters`
- optional `model_metadata` for `eml_tree`

## Development

문법 확인:

```bash
python3 -m py_compile eml_circuit/__init__.py \
  eml_circuit/benchmarks.py \
  eml_circuit/training.py \
  examples/run_emlstack_demo.py \
  examples/run_benchmark_suite.py
```

테스트:

```bash
python3 -m unittest -q
```

이 환경에서 `torch`가 없으면 테스트는 skip 또는 import error가 날 수 있습니다.
