# Research Notes

이 문서는 EMLStack의 개념 메모와 벤치마크 설계 의도를 정리한 문서입니다. 빠른 실행 방법은 [README.md](/mnt/d/GitHub/Eml_Circuit/README.md)를 보십시오.

## EML Operator

기본 EML 연산자는 다음과 같습니다.

$$
\mathrm{eml}(x,y)=\exp(x)-\ln(y)
$$

이 저장소는 EML을 단순 pointwise activation으로 쓰기보다, ordered pair를 받아 계산하는 circuit primitive로 다룹니다.

## EMLStack Block

현재 구현은 다음 형태의 residual block을 사용합니다.

$$
Z=\mathrm{RMSNorm}(H)
$$

$$
[U,V]=W_p Z+b
$$

$$
U_{\mathrm{safe}}=\frac{U}{1+|U|/c}
$$

$$
V_{\mathrm{pos}}=\mathrm{softplus}(V)+\epsilon
$$

$$
E=\exp(U_{\mathrm{safe}})-\ln(V_{\mathrm{pos}})
$$

$$
H' = H + \lambda W_o E
$$

실제 구현 위치:

- [eml_circuit/layers.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/layers.py)
- [eml_circuit/functional.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/functional.py)

## Why Residual EML

핵심 의도는 다음입니다.

- EML tree search가 어려운 함수를 residual circuit learning으로 다루기
- intermediate expression reuse를 hidden channel에 저장하기
- raw `exp` / `log` branch의 수치 불안정을 softsign clamp와 positive log branch로 완화하기

## Benchmark Split

벤치마크는 두 그룹으로 나뉩니다.

### `tree`

`eml_tree` baseline과 직접 비교하기 위한 그룹입니다.

- `tree_shared_subexpr_a`
  shared subexpression reuse가 많은 EML-structured target
- `tree_deep_chain_b`
  깊은 EML chain을 요구하는 target
- `tree_circuit_reuse_c`
  low-depth, high-reuse circuit target

이 그룹은 `EMLStack vs EML tree` 비교를 의도합니다.

### `mlp`

`mlp` baseline과 비교하기 위한 그룹입니다.

- `mlp_gelu_mix_a`
- `mlp_relu_piecewise_b`
- `mlp_silu_gate_c`

이 그룹은 타깃 함수 자체가 `GELU`, `ReLU`, `SiLU`를 포함하므로 `EMLStack vs MLP` 비교를 의도합니다.

## Tree Baseline

현재 `eml_tree`는 raw EML tree를 beam search로 확장하고, 선택된 tree feature 위에 선형 readout을 얹는 baseline입니다.

구현 위치:

- [eml_circuit/symbolic.py](/mnt/d/GitHub/Eml_Circuit/eml_circuit/symbolic.py)

이 구현은 실험용 baseline이며, 외부 symbolic regression 시스템을 그대로 복제한 것은 아닙니다.

## Evaluation

현재 기본 지표:

- `train_mse`
- `extrap_mse`

향후 확장 후보:

- multi-seed aggregate
- benchmark별 CSV export
- symbolic simplification / snap 이후 오차
- wall-clock / search cost 비교
