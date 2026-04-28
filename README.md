# EMLStack: Residual EML Circuit Layer

## 1. 출발점

EML 논문의 기본 연산자는

[
\operatorname{eml}(x,y)=\exp(x)-\ln(y)
]

입니다.

논문은 이 단일 이항 연산자와 상수 (1)만으로 exp, log, 사칙연산, 거듭제곱, 삼각함수 등 표준 elementary function을 구성할 수 있다고 주장합니다. 또한 모든 EML 표현은 동일한 binary node로 이루어진 tree가 되며, 문법은 (S\to1\mid\operatorname{eml}(S,S))처럼 단순해집니다. ([arXiv][1])

이걸 딥러닝으로 옮길 때 중요한 점은:

[
\text{Linear} \rightarrow \text{Activation}
]

구조가 아니라,

[
\text{Linear} \rightarrow (u,v) \rightarrow \operatorname{EML}(u,v)
]

구조로 가야 한다는 것입니다.

즉 선형층은 feature를 직접 만드는 게 아니라, **EML node에 들어갈 ordered pair**를 생성합니다.

---

## 2. 기본 블록

입력 hidden state를 (H_l)라고 하면, 한 EMLStack block은 다음처럼 정의합니다.

[
Z_l = \operatorname{RMSNorm}(H_l)
]

[
[U_l,V_l] = W_p Z_l + b
]

여기서 (W_p)는 출력 차원이 (2m)인 선형층입니다.
그 결과를 두 채널로 나눕니다.

[
U_l,V_l\in\mathbb{R}^{m}
]

그리고 각 순서쌍 ((U_j,V_j))를 EML에 넣습니다.

[
E_l
===

\operatorname{EML}(U_l,V_l)
]

다만 순정 EML은 수치적으로 위험하므로 안정화된 형태를 씁니다.

[
U_{\text{safe}}
===============

\frac{U_l}{1+|U_l|/c}
]

[
V_{\text{pos}}
==============

\operatorname{softplus}(V_l)+\epsilon
]

[
E_l
===

## \exp(U_{\text{safe}})

\ln(V_{\text{pos}})
]

최종적으로 residual을 붙입니다.

[
H_{l+1}
=======

H_l
+
\lambda_l W_o E_l
]

전체 블록은:

[
\boxed{
H_{l+1}
=======

H_l
+
\lambda_l W_o
\left[
\exp\left(
\frac{U_l}{1+|U_l|/c}
\right)
-------

\ln\left(
\operatorname{softplus}(V_l)+\epsilon
\right)
\right]
}
]

[
[U_l,V_l]=W_p\operatorname{RMSNorm}(H_l)+b
]

---

## 3. 왜 activation이 필요 없는가

EML 자체가 이미 강한 비선형 연산입니다.

[
\operatorname{eml}(u,v)=e^u-\ln v
]

여기에는 exponential branch와 logarithmic branch가 모두 들어 있습니다.
따라서 ReLU, GELU, SiLU 같은 pointwise activation을 추가하면 구조가 중복됩니다.

이 구조에서 비선형성은 activation이 아니라 **EML node primitive**에서 발생합니다.

정리하면:

| 기존 MLP                    | EMLStack                   |
| ------------------------- | -------------------------- |
| Linear가 hidden feature 생성 | Linear가 ((u,v)) pair 생성    |
| Activation이 비선형성 제공       | EML이 비선형성 제공               |
| feature는 dense vector     | feature는 EML circuit state |
| 해석성 낮음                    | 중간 EML node를 수식 후보로 해석 가능  |

---

## 4. 왜 residual이 중요한가

residual은 선택사항이 아니라 거의 필수입니다.

순수하게

[
H_{l+1}=\operatorname{EMLBlock}(H_l)
]

로 두면 매 layer가 representation 전체를 덮어씁니다. EML은 (e^u) 때문에 쉽게 폭주하고, (-\ln v) 때문에 (v\to0) 근방에서 불안정해집니다.

논문에서도 EML net 학습 중 다중 합성된 exponential로 인한 overflow와 NaN 문제가 발생했고, exp argument/value clamping이 필요했다고 보고합니다. 또 random initialization에서 exact recovery 성공률이 depth 2에서는 100%, depth 3–4에서는 약 25%, depth 5에서는 1% 미만, depth 6에서는 448회 중 성공 0이었다고 합니다. 즉 깊은 EML tree 탐색은 구조적으로 어렵습니다. ([arXiv][1])

residual을 쓰면 EML block은 전체 표현을 갈아엎는 것이 아니라,

[
H_{l+1}=H_l+\lambda_l\Delta H_l
]

처럼 작은 correction을 추가합니다.

이게 중요한 이유는:

1. identity path가 살아 있어 학습이 덜 터짐
2. gradient highway가 생김
3. EML node가 feature constructor로 작동함
4. hardening/snap 과정에서 성능 붕괴가 줄어듦
5. EML tree가 아니라 EML circuit/DAG처럼 중간 feature를 누적 가능

---

## 5. softsign clamp를 쓰는 이유

exp branch는 반드시 제어해야 합니다.

처음에는

[
U_{\text{safe}}=c\tanh(U/c)
]

를 생각할 수 있지만, `tanh`는 포화가 너무 빠릅니다. 대신 softsign clamp가 더 낫습니다.

[
U_{\text{safe}}
===============

\frac{U}{1+|U|/c}
]

이 방식은 (U_{\text{safe}}\in(-c,c))로 exp 입력을 제한하면서도, gradient가 `tanh`보다 천천히 죽습니다.

| 방식             | 장점         | 단점                     |
| -------------- | ---------- | ---------------------- |
| no clamp       | 표현력 최대     | exp 폭주                 |
| hard clamp     | 폭주 방지      | 경계에서 gradient 끊김       |
| tanh clamp     | 부드러움       | 포화 빠름                  |
| softsign clamp | 부드럽고 포화 느림 | 여전히 큰 입력에서 gradient 감소 |

초기값은 보수적으로:

[
c=3\sim5
]

정도가 적절합니다.

---

## 6. 연구 기여 포인트

단순히 “EML을 neural layer로 만들었다”만으로는 약합니다.
기여가 되려면 기존 EML tree search가 어려운 함수군을 명확히 보여야 합니다.

가장 강한 포인트는:

[
\boxed{
\text{tree search} \rightarrow \text{circuit learning}
}
]

입니다.

기존 EML tree는 중간식을 재사용하지 못합니다. 같은 subexpression이 여러 번 필요하면 tree 안에서 계속 복사해야 합니다.

반면 EMLStack은 channel에 중간 feature를 저장하고, 다음 layer에서 재사용할 수 있습니다. 즉 tree가 아니라 DAG/circuit에 가깝습니다.

따라서 핵심 claim은 다음과 같습니다.

> EMLStack converts EML symbolic regression from deep tree search into residual circuit learning with reusable intermediate features.

또는 더 짧게:

> EMLStack learns tree-hard but circuit-easy elementary functions.

---

## 7. 만들어야 할 benchmark

기여를 보이려면 다음 함수군이 필요합니다.

### A. Shared subexpression family

중간식을 하나 정의합니다.

[
g(x,y)=\operatorname{eml}(x,y)
]

그다음 이 (g)를 여러 번 재사용하는 함수를 만듭니다.

[
f_k(x,y)
========

\sum_{i=1}^{k}
a_i
\left[
\exp(\alpha_i g(x,y))
---------------------

\ln(\beta_i+\gamma_i g(x,y)^2)
\right]
]

여기서 기존 tree는 (g(x,y))를 여러 branch에 반복 복사해야 합니다.
EMLStack은 첫 layer에서 (g)를 만들고, 다음 layer에서 재사용할 수 있습니다.

이게 가장 강한 benchmark입니다.

### B. Deep EML chain

[
g_0(x,y)=x
]

[
g_{t+1}(x,y)
============

\operatorname{eml}(s(g_t),y)
]

[
s(z)=\frac{z}{1+|z|/c}
]

[
f_L(x,y)=g_L(x,y)
]

이건 기존 EML tree의 depth 한계를 직접 찌르는 benchmark입니다.
다만 모델 구조와 target generator가 너무 비슷해 보일 수 있으므로 보조 실험으로 두는 편이 낫습니다.

### C. Low-depth circuit, high-size tree

예를 들어:

[
z_1=\operatorname{eml}(x,y)
]

[
z_2=\operatorname{eml}(z_1,1)
]

[
z_3=\operatorname{eml}(1,z_1)
]

[
z_4=\operatorname{eml}(z_2,z_3)
]

[
f(x,y)=w_1z_1+w_2z_2+w_3z_3+w_4z_4
]

EMLStack에서는 (z_1,z_2,z_3,z_4)가 channel로 남습니다.
하지만 pure tree는 중간식을 재사용하지 못해서 expression size가 커집니다.

---

## 8. 평가 지표

train MSE만 보면 안 됩니다. MLP도 근사는 잘합니다.

핵심 지표는 다음입니다.

| 지표                  | 의미                          |
| ------------------- | --------------------------- |
| interpolation MSE   | 학습 범위 안에서 맞는가               |
| extrapolation MSE   | 범위 밖에서도 맞는가                 |
| snap 후 MSE          | symbolic hardening 이후 유지되는가 |
| recovery rate       | seed별 성공률                   |
| NaN/overflow rate   | 수치적으로 안정적인가                 |
| effective tree size | 같은 함수를 tree로 표현할 때 얼마나 커지는가 |
| circuit depth/width | EMLStack에서 얼마나 작게 표현되는가     |

가장 중요한 것은:

[
\boxed{
\text{snap 후 extrapolation MSE}
}
]

입니다.

---

## 9. PyTorch 기본형

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


def softsign_clip(x, c):
    return x / (1.0 + x.abs() / c)


class EMLResidualBlock(nn.Module):
    def __init__(
        self,
        dim,
        width=None,
        c=5.0,
        eps=1e-4,
        init_scale=1e-3,
    ):
        super().__init__()
        width = width or dim

        self.norm = RMSNorm(dim)
        self.pair_proj = nn.Linear(dim, 2 * width)
        self.out_proj = nn.Linear(width, dim)

        self.c = c
        self.eps = eps
        self.res_scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, h):
        z = self.norm(h)

        pair = self.pair_proj(z)
        u, v = pair.chunk(2, dim=-1)

        u_safe = softsign_clip(u, self.c)
        v_pos = F.softplus(v) + self.eps

        e = torch.exp(u_safe) - torch.log(v_pos)

        return h + self.res_scale * self.out_proj(e)
```

---

## 10. 한 줄 정리

이 아이디어의 정확한 포지션은 이것입니다.

[
\boxed{
\text{EMLStack은 activation replacement가 아니라, trainable elementary-function circuit layer다.}
}
]

핵심 설계는:

[
\boxed{
\text{RMSNorm}
\rightarrow
\text{Linear}_{2m}
\rightarrow
(U,V)
\rightarrow
\text{softsign-stabilized EML}
\rightarrow
\text{small residual update}
}
]

핵심 기여는:

[
\boxed{
\text{기존 EML tree search가 어려운 subexpression-sharing 함수를 EML circuit으로 학습한다.}
}
]


[1]: https://arxiv.org/pdf/2603.21852v2 "All elementary functions from a single binary operator"
