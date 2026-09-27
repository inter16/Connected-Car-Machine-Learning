가능합니다. 핵심은 완성 코드를 먼저 외우는 것이 아니라, **“환경을 표현한다 → 상태 전이를 만든다 → Bellman update를 구현한다 → 실행 스크립트를 만든다”** 순서로 직접 쌓아가는 것입니다. 첨부 과제의 1번은 Policy Evaluation, 2번은 Value Iteration이고, 둘 다 4×4 GridWorld를 사용합니다.  

아래 순서대로 실제로 빈 파일에서 한 줄씩 작성하면, 제가 앞서 만든 코드와 같은 구조와 동작에 도달합니다.

1. **먼저 프로젝트 뼈대만 만든다.** 처음부터 알고리즘을 쓰지 말고 역할별 파일을 나눕니다.

```text
assignment2-pytorch/
├── pyproject.toml
├── .gitignore
├── README.md
│
├── src/
│   └── connected_car_rl/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   └── gridworld.py
│       └── algorithms/
│           ├── __init__.py
│           └── dynamic_programming.py
│
├── scripts/
│   ├── policy_evaluation.py
│   └── value_iteration.py
│
└── tests/
    └── test_dynamic_programming.py
```

이 구조에서 `envs/`는 **문제가 어떻게 생겼는지**, `algorithms/`는 **그 문제를 어떻게 푸는지**, `scripts/`는 **실제로 무엇을 실행할지**를 담당합니다.

처음 손으로 구현할 때 가장 중요한 원칙은 `policy_evaluation.py`부터 시작하지 않는 것입니다. 아직 환경이 없기 때문입니다. 따라서 가장 먼저 `gridworld.py`를 완성합니다.

`src/connected_car_rl/envs/gridworld.py`를 열고 필요한 라이브러리부터 적습니다.

```python
from __future__ import annotations

from dataclasses import dataclass, field

import torch
```

여기서 PyTorch는 상태 가치, 보상, 정책 등을 `torch.Tensor`로 저장하기 위해 사용합니다. `dataclass`는 GridWorld의 설정값을 깔끔하게 묶기 위한 Python 기능입니다.

이제 환경 클래스의 외형부터 작성합니다.

```python
@dataclass
class GridWorld:
    rows: int = 4
    cols: int = 4
    initial_state: int = 0
    terminal_state: int = 15
    step_reward: float = -1.0
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float64
```

여기까지 작성하면 아직 GridWorld가 움직이지는 않습니다. 단지 “4×4 격자이고, 시작 상태는 0, terminal은 15이며, 한 번 이동할 때 보상은 -1이다”라는 설정만 저장한 상태입니다.

그다음 행동을 정의합니다. 행동 순서는 반드시 일관되기만 하면 되지만, 저는 `North, South, East, West` 순서로 만들었습니다.

```python
    action_deltas: tuple[tuple[int, int], ...] = field(
        default=((-1, 0), (1, 0), (0, 1), (0, -1)),
        init=False,
    )
```

각 튜플은 `(행 변화량, 열 변화량)`입니다. 예를 들어 `(-1, 0)`은 한 행 위로 가므로 North입니다.

여기서 머릿속으로 다음과 같이 연결하면 됩니다.

```text
0 -> North -> (-1, 0)
1 -> South -> (+1, 0)
2 -> East  -> (0, +1)
3 -> West  -> (0, -1)
```

이제 객체가 만들어졌을 때 자동으로 필요한 데이터를 준비하도록 `__post_init__()`을 만듭니다.

```python
    def __post_init__(self) -> None:
        self.device = torch.device(self.device)

        self.num_states = self.rows * self.cols
        self.num_actions = len(self.action_deltas)
```

4×4이므로 `num_states`는 16이고, 행동이 네 가지이므로 `num_actions`는 4가 됩니다.

여기서부터가 환경 구현의 핵심입니다. 각 상태에서 각각의 행동을 했을 때 **다음 상태가 무엇인지 미리 표로 만들어두는 방식**을 사용합니다.

```python
        self.next_state = self._build_next_state_table()
```

아직 `_build_next_state_table()`을 만들지 않았으므로 잠시 그대로 둡니다.

다음으로 보상 표를 만듭니다.

```python
        self.reward = torch.full(
            (self.num_states, self.num_actions),
            self.step_reward,
            dtype=self.dtype,
            device=self.device,
        )
```

여기서 `reward`의 shape은 다음과 같습니다.

```text
[16, 4]
```

즉 각 행은 상태 하나, 각 열은 행동 하나입니다.

처음에는 모든 행동의 보상을 -1로 채웁니다.

```text
s0 : [-1, -1, -1, -1]
s1 : [-1, -1, -1, -1]
...
s15: [-1, -1, -1, -1]
```

하지만 `s15`는 terminal state이므로 그 상태에 도착한 뒤에는 더 이상 보상을 받아서는 안 됩니다. 따라서 다음을 추가합니다.

```python
        self.reward[self.terminal_state, :] = 0.0
```

이제 terminal state를 쉽게 구별할 수 있도록 mask도 만듭니다.

```python
        self.terminal_mask = torch.zeros(
            self.num_states,
            dtype=torch.bool,
            device=self.device,
        )

        self.terminal_mask[self.terminal_state] = True
```

여기까지 끝냈으면 이제 가장 중요한 `_build_next_state_table()`을 작성합니다.

먼저 함수와 빈 tensor를 만듭니다.

```python
    def _build_next_state_table(self) -> torch.Tensor:
        table = torch.empty(
            (self.num_states, self.num_actions),
            dtype=torch.long,
            device=self.device,
        )
```

이 `table` 역시 shape이 `[16, 4]`입니다. 차이는 reward가 아니라 **다음 state 번호**를 저장한다는 것입니다.

예를 들어 `s0`에서 South를 하면 `s4`이므로 대략 다음과 같은 정보가 들어갑니다.

```text
state 0:
North -> 0
South -> 4
East  -> 1
West  -> 0
```

격자 밖으로 나가려고 하면 현재 상태에 그대로 있도록 구현합니다.

이제 모든 state를 순회합니다.

```python
        for state in range(self.num_states):
```

terminal state라면 어디로 움직여도 계속 terminal state에 있도록 합니다.

```python
            if state == self.terminal_state:
                table[state, :] = state
                continue
```

그다음 state 번호를 격자의 row와 column으로 바꿉니다.

```python
            row, col = divmod(state, self.cols)
```

예를 들어 `state=6`이면 4열이므로 row는 1, col은 2가 됩니다.

이제 네 행동을 하나씩 확인합니다.

```python
            for action, (d_row, d_col) in enumerate(
                self.action_deltas
            ):
```

현재 위치에 행동 방향을 더합니다.

```python
                next_row = row + d_row
                next_col = col + d_col
```

그런데 North를 했더니 row가 -1이 될 수도 있습니다. 따라서 격자 범위를 넘어가지 못하도록 제한합니다.

```python
                next_row = min(
                    max(next_row, 0),
                    self.rows - 1,
                )

                next_col = min(
                    max(next_col, 0),
                    self.cols - 1,
                )
```

이제 다시 row와 column을 state 번호로 바꿉니다.

```python
                next_state = (
                    next_row * self.cols + next_col
                )
```

그리고 table에 저장합니다.

```python
                table[state, action] = next_state
```

모든 상태를 처리했으면 반환합니다.

```python
        return table
```

여기까지 작성했으면 `_build_next_state_table()` 전체는 다음 논리를 갖습니다.

```text
현재 state
    ↓
(row, col)로 변환
    ↓
행동에 따라 row/col 변경
    ↓
격자 바깥이면 경계에서 멈춤
    ↓
다음 state 번호 계산
    ↓
next_state table에 저장
```

이 시점에서 환경의 가장 어려운 부분은 끝났습니다.

다음으로 Policy Evaluation에 필요한 **uniform random policy**를 만듭니다. 과제 1에서는 네 행동을 동일 확률로 선택한다고 했으므로 각 행동 확률은 0.25입니다.

```python
    def uniform_random_policy(self) -> torch.Tensor:
        return torch.full(
            (self.num_states, self.num_actions),
            1.0 / self.num_actions,
            dtype=self.dtype,
            device=self.device,
        )
```

결과는 대략 다음과 같습니다.

```text
s0  -> [0.25, 0.25, 0.25, 0.25]
s1  -> [0.25, 0.25, 0.25, 0.25]
...
s15 -> [0.25, 0.25, 0.25, 0.25]
```

마지막으로 결과 출력을 쉽게 하기 위한 함수 하나를 작성합니다.

```python
    def as_grid(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return values.reshape(self.rows, self.cols)
```

원래 value tensor는 `[16]`인데 과제에서는 4×4 matrix로 출력해야 하므로 `[4, 4]`로 바꾸는 것입니다.

여기까지 작성했으면 `gridworld.py`가 끝납니다.

---

이제 두 번째 파일인

```text
src/connected_car_rl/algorithms/dynamic_programming.py
```

를 작성합니다.

먼저 import합니다.

```python
import torch

from connected_car_rl.envs.gridworld import GridWorld
```

여기서부터 **1번 Policy Evaluation**을 구현합니다.

Policy Evaluation이 무엇을 계산하려는지 먼저 이해해야 코드가 외워지지 않습니다.

정책이 정해져 있을 때 상태 가치의 반복 계산은 다음 식을 사용합니다.

$$
V_{k+1}(s)
=
\sum_a
\pi(a\mid s)
\left[
r(s,a)
+
\gamma V_k(s')
\right]
$$

```latex
V_{k+1}(s)
=
\sum_a
\pi(a\mid s)
\left[
r(s,a)
+
\gamma V_k(s')
\right]
```

말로 풀면 다음 뜻입니다.

```text
현재 state s에서
    ↓
가능한 각 action a를 생각하고
    ↓
그 action의 즉시 보상
+
다음 state의 기존 가치
    ↓
각 action을 선택할 확률로 가중평균
    ↓
새로운 V(s)
```

이 구조를 그대로 Python으로 옮깁니다.

함수 틀부터 작성합니다.

```python
def policy_evaluation(
    env: GridWorld,
    *,
    gamma: float = 1.0,
    iterations: int = 1000,
    initial_value: float = 0.0,
    policy: torch.Tensor | None = None,
) -> torch.Tensor:
```

`env`는 우리가 만든 GridWorld이고, `gamma`는 discount factor, `iterations`는 과제에서 요구하는 1000회입니다.

정책을 따로 주지 않았다면 uniform random policy를 사용합니다.

```python
    if policy is None:
        policy = env.uniform_random_policy()
```

이제 각 state의 초기 value를 만듭니다.

```python
    values = torch.full(
        (env.num_states,),
        float(initial_value),
        dtype=env.dtype,
        device=env.device,
    )
```

과제 1-3에서 `initial_value=0`과 `initial_value=10`을 비교하므로 이 값을 매개변수로 만든 것입니다.

terminal은 항상 0으로 둡니다.

```python
    values[env.terminal_state] = 0.0
```

이제 1000번 반복합니다.

```python
    for _ in range(iterations):
```

반복마다 **이전 iteration의 값**을 기준으로 새로운 값을 계산합니다.

```python
        old_values = values
```

그다음 처음으로 Bellman 식을 tensor 코드로 옮깁니다.

```python
        q_values = (
            env.reward
            + gamma * old_values[env.next_state]
        )
```

처음 보면 가장 이해하기 어려운 줄입니다.

`env.next_state`의 shape은 다음과 같습니다.

```text
[16, 4]
```

그러므로

```python
old_values[env.next_state]
```

를 하면 각 state와 action에 대한 **다음 state의 value**가 한꺼번에 나옵니다.

shape은 그대로

```text
[16, 4]
```

입니다.

따라서

```python
env.reward
```

도 `[16,4]`,

```python
old_values[env.next_state]
```

도 `[16,4]`이므로 서로 바로 더할 수 있습니다.

즉 `q_values[state, action]`은 “그 state에서 그 action을 했을 때 얻는 값”입니다.

Policy Evaluation에서는 하나의 action을 고르는 것이 아니라 네 action을 random policy에 따라 평균냅니다.

```python
        new_values = torch.sum(
            policy * q_values,
            dim=1,
        )
```

여기서 `policy * q_values`를 하면 각 action의 값에 각각 0.25가 곱해지고, `dim=1`로 더하면 행동 방향에 대해 합산됩니다.

예를 들어 어떤 상태에서 네 행동 값이 다음이라고 가정해보겠습니다.

```text
[-5, -3, -4, -4]
```

정책은

```text
[0.25, 0.25, 0.25, 0.25]
```

이므로 결국 네 값의 평균이 그 state의 새로운 가치가 됩니다.

그다음 terminal value를 다시 0으로 고정합니다.

```python
        new_values[env.terminal_state] = 0.0
```

그리고 이번 결과를 다음 iteration의 기준으로 넘깁니다.

```python
        values = new_values
```

1000번이 끝나면 반환합니다.

```python
    return values
```

Policy Evaluation 구현의 핵심을 코드 없이 다시 생각하면 다음입니다.

```text
1. V(s)를 초기화한다.
2. 모든 행동의 [reward + 다음 state value]를 계산한다.
3. 정책 확률을 곱한다.
4. 행동 방향으로 합한다.
5. 결과를 새로운 V(s)로 둔다.
6. 1000번 반복한다.
```

---

이제 같은 파일에 **2번 Value Iteration**을 작성합니다.

여기서는 정책이 없습니다. 대신 항상 가장 가치가 큰 action을 선택한다고 생각합니다.

Bellman optimality update는 다음입니다.

$$
V_{k+1}(s)
=
\max_a
\left[
r(s,a)
+
\gamma V_k(s')
\right]
$$

```latex
V_{k+1}(s)
=
\max_a
\left[
r(s,a)
+
\gamma V_k(s')
\right]
```

Policy Evaluation과 비교하면 거의 같습니다.

Policy Evaluation:

```text
네 action의 값을 정책 확률에 따라 평균
```

Value Iteration:

```text
네 action의 값 중 가장 큰 것 선택
```

따라서 함수부터 만듭니다.

```python
def value_iteration(
    env: GridWorld,
    *,
    gamma: float = 1.0,
    iterations: int = 1000,
    initial_value: float = 0.0,
) -> torch.Tensor:
```

초기 value도 같습니다.

```python
    values = torch.full(
        (env.num_states,),
        float(initial_value),
        dtype=env.dtype,
        device=env.device,
    )

    values[env.terminal_state] = 0.0
```

반복도 같습니다.

```python
    for _ in range(iterations):
        old_values = values
```

각 action의 값까지 구하는 것도 완전히 같습니다.

```python
        q_values = (
            env.reward
            + gamma * old_values[env.next_state]
        )
```

오직 그다음 줄만 달라집니다.

```python
        new_values = torch.max(
            q_values,
            dim=1,
        ).values
```

왜 `dim=1`인지 반드시 이해해야 합니다.

`q_values`의 구조가 다음과 같기 때문입니다.

```text
                 Action
             N    S    E    W
State s0    ...
State s1    ...
State s2    ...
...
```

즉 `dim=1`은 한 state 안에서 네 행동을 비교하라는 뜻입니다.

따라서 각 행에서 max를 뽑으면 state마다 하나의 최적 value가 나옵니다.

마지막은 Policy Evaluation과 같습니다.

```python
        new_values[env.terminal_state] = 0.0
        values = new_values

    return values
```

여기까지 오면 알고리즘 구현이 모두 끝납니다.

---

이제 실제 실행 파일을 만듭니다.

먼저

```text
scripts/policy_evaluation.py
```

를 작성합니다.

처음에는 복잡한 argument parser 없이 최소 코드로 시작하는 것을 권합니다.

```python
from connected_car_rl.envs.gridworld import GridWorld
from connected_car_rl.algorithms.dynamic_programming import (
    policy_evaluation,
)
```

환경을 만듭니다.

```python
env = GridWorld()
```

알고리즘을 실행합니다.

```python
values = policy_evaluation(
    env,
    gamma=1.0,
    iterations=1000,
    initial_value=0.0,
)
```

4×4로 출력합니다.

```python
print(env.as_grid(values))
```

처음에는 여기까지만 작성해서 실행해 보는 것이 좋습니다.

정상이라면 다음 형태가 나와야 합니다.

```text
tensor([[-59.4286, -57.4286, -54.2857, -51.7143],
        [-57.4286, -54.5714, -49.7143, -45.1429],
        [-54.2857, -49.7143, -40.8571, -30.0000],
        [-51.7143, -45.1429, -30.0000,   0.0000]],
       dtype=torch.float64)
```

그러고 나서 과제 1-3 확인을 위해 다음처럼 바꿔봅니다.

```python
initial_value=10.0
```

1000번 반복한 뒤 동일한 값으로 수렴하는지 확인하면 됩니다.

---

다음으로

```text
scripts/value_iteration.py
```

를 만듭니다.

```python
from connected_car_rl.envs.gridworld import GridWorld
from connected_car_rl.algorithms.dynamic_programming import (
    value_iteration,
)

env = GridWorld()

values = value_iteration(
    env,
    gamma=1.0,
    iterations=1000,
    initial_value=0.0,
)

print(env.as_grid(values))
```

정상적으로 구현했다면 다음이 나옵니다.

```text
tensor([[-6., -5., -4., -3.],
        [-5., -4., -3., -2.],
        [-4., -3., -2., -1.],
        [-3., -2., -1.,  0.]],
       dtype=torch.float64)
```

여기까지 동작한 뒤에야 `argparse` 같은 편의 기능을 추가하는 것이 좋습니다. 처음부터 명령행 옵션까지 넣으면 강화학습 알고리즘보다 Python 주변 코드 때문에 흐름을 놓치기 쉽습니다.

---

손으로 구현할 때는 특히 **tensor shape을 항상 머릿속에 적어두는 것**을 권합니다. 이 과제에서 핵심 tensor는 사실 네 개뿐입니다.

```text
values
shape = [16]

next_state
shape = [16, 4]

reward
shape = [16, 4]

policy
shape = [16, 4]
```

그리고 계산 흐름은 다음 하나로 압축할 수 있습니다.

```text
values [16]
       │
       │ next_state로 indexing
       ▼
next state values [16,4]
       │
       │ reward 더하기
       ▼
q_values [16,4]
       │
       ├── Policy Evaluation
       │      policy를 곱하고 sum
       │
       └── Value Iteration
              max
       ▼
new_values [16]
```

이 구조를 이해하면 `policy_evaluation()`과 `value_iteration()`은 사실 마지막 연산 하나만 다르다는 것을 알 수 있습니다.

특히 직접 다시 구현할 때는 다음 순서가 가장 좋습니다. **첫날에는 `GridWorld`만 보고 처음부터 다시 작성하고 실행 → 그다음 `policy_evaluation()`만 작성 → 마지막으로 `value_iteration()`을 Policy Evaluation에서 변형**하십시오. Value Iteration을 별개의 새로운 알고리즘처럼 외우기보다는, “정책에 따른 평균을 최댓값 선택으로 바꾼 것”으로 이해하는 편이 훨씬 쉽습니다.

그리고 현재 구현에서 제가 의도적으로 사용하지 않은 것도 중요합니다. `torch.nn.Module`, `optimizer`, `loss.backward()`는 전혀 필요하지 않습니다. 이번 과제의 Policy Evaluation과 Value Iteration은 **신경망 학습이 아니라 이미 알려진 상태 전이 모델에 Bellman 식을 반복 적용하는 동적 계획법(dynamic programming)**이기 때문입니다. PyTorch는 여기서 신경망 프레임워크라기보다 tensor 계산 도구로 사용하고 있습니다.

원한다면 다음 단계로는 **`gridworld.py`부터 시작해서 “지금 이 한 줄을 왜 쓰는가?”까지 코드 한 줄 단위로 설명하는 실습형 튜토리얼** 형태로 진행할 수 있습니다. 그 방식이면 제가 먼저 정답 코드를 보여주지 않고, 작성해야 할 다음 3~5줄만 제시하면서 실제로 직접 구현하는 순서대로 진행할 수 있습니다.
