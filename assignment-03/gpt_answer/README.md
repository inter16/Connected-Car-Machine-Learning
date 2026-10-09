# HW3: Grid World — Every-Visit MC / TD(0) (PyTorch)

첨부 과제의 **예제 1(Every-Visit Monte Carlo)**과 **예제 2(TD(0))**를 하나의 4×4 Grid World 환경으로 구현했습니다. 실행 후 가치함수를 4행 4열로 출력하고, CSV 및 PNG로 저장합니다. `REPORT_ANSWERS.md`에는 2-3번의 서술형 답안을 포함했습니다.

## 1. 과제 조건 및 검증된 환경 규칙

강의자료 **`Lecture 3: Model-Free Prediction`의 8~10, 12, 14~15페이지**, **`Lecture 2: Planning by Dynamic Programming`의 9~11, 14페이지**, 과제 `HW3.pdf`의 1~2페이지를 대조했습니다. 학습률·정책·격자 경계 처리 규칙은 강의자료에 근거합니다.

| 항목 | 설정 |
|---|---|
| 환경 | 4×4, 상태 S0~S15 |
| 시작 / 종료 | S0 / S15 |
| 행동 / 정책 | 상·하·좌·우 중 동일 확률(각 1/4)로 선택 |
| 보상 | 매 행동마다 -1 |
| 할인율 | `gamma=1.0` |
| 학습량 | 기본값 `50,000`개의 **완료된 에피소드** |
| MC 초기값 | `N(s)=0`, `C(s)=0` (강의자료 p.8) |
| TD(0) 초기값 | `V(s)=0` (강의자료 p.15) |
| 상태 전이 | 같은 상태와 행동에 대해 다음 상태가 결정됨 (강의자료 p.8, 15) |
| 경계 처리 | 격자를 벗어나는 방향을 선택하면 제자리에 머물고 -1을 받음 (Lecture 2 p.11) |
| TD(0) 학습률 | **`alpha=0.01` (강의자료 p.15에 명시, 변경 가능)** |
| 재현성 | `seed=42` (재현 가능한 결과를 위한 구현 설정; 강의자료의 지정값은 아님) |

**남은 해석상 불확실성:** `HW3.pdf`의 `50,000 iterations`가 정확히 50,000개 에피소드인지 50,000번 환경 행동인지는 세 PDF에서 확인되지 않습니다. 현재 코드는 **완료된 에피소드 50,000개**로 해석합니다. Lecture 2 p.14의 `k=1000`은 *DP 정책 평가의 반복 횟수*이므로 MC·TD(0)의 반복 단위를 확정하는 근거가 아닙니다. 과제 PDF에는 MATLAB 시작 코드가 없어 원본 코드를 번역한 것이 아니라 문제 조건대로 새로 구현했습니다.

**서로 다른 Grid World의 구별:** Lecture 2 p.9~11은 `S0`와 `S15`가 모두 종료 상태인 *별개 예제*이지만, Lecture 2 p.14와 Lecture 3 및 HW3의 Grid World는 `S15`만 종료 상태입니다. 이 프로젝트에서는 `S0`를 **출발 상태이자 일반(비종료) 상태**로 처리합니다. Lecture 2 p.14의 무작위 정책 가치표(`-59, -57, ..., 0`)는 프로젝트의 벨만 방정식 기준 계산 결과와 반올림 수준에서 일치합니다.

출처별 근거와 확인되지 않은 항목은 [`SOURCE_CONDITIONS.md`](SOURCE_CONDITIONS.md)에서 별도로 정리했습니다.

## 2. 프로젝트 폴더 구조

```text
gridworld-hw3-pytorch/
├── main.py                      # 명령행 실행 진입점
├── pyproject.toml               # Python 패키지 설정 / 의존성
├── README.md
├── REPORT_ANSWERS.md            # 1-1 ~ 2-3 설명 / 서술형 답안
├── SOURCE_CONDITIONS.md         # 강의자료의 근거 및 남은 불확실성
├── CODE_WALKTHROUGH_KO.md       # 직접 작성할 때의 코드 구현 순서
├── src/
│   └── gridworld_rl/
│       ├── cli.py                # 학습 실행, 결과 파일 저장
│       ├── envs/
│       │   └── gridworld.py     # 공통 환경: reset(), step()
│       ├── agents/
│       │   ├── every_visit_mc.py # 예제 1
│       │   └── td0.py           # 예제 2
│       └── utils/
│           ├── analysis.py      # 벨만 기대 방정식 정확해 비교
│           └── output.py        # 4×4 표 / CSV / PNG
├── tests/
│   ├── test_env.py
│   └── test_agents.py
└── results/
    ├── terminal_output.txt      # 전체 실행 출력
    ├── mc_values.csv
    ├── mc_values.png             # 예제 1 결과
    ├── td0_values.csv
    ├── td0_values.png            # 예제 2 결과
    └── exact_values.csv         # 기준값 (학습 결과 아님)
```

여기서 `envs/`는 환경, `agents/`는 학습 알고리즘, `utils/`는 결과 처리·검증, `tests/`는 자동 검증을 담당합니다. 신경망을 사용하지 않는 **표 형태의 상태가치 추정**이므로 `models/` 폴더나 `torch.nn.Module`은 필요하지 않습니다.

## 3. 설치

Python 3.12를 사용하는 Linux/macOS 터미널 기준:

```bash
# 프로젝트 압축 해제 후 해당 폴더로 이동
cd gridworld-hw3-pytorch

# 가상환경 생성·활성화 (pyenv 등 기존 환경을 활용해도 됩니다)
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

이미 PyTorch를 설치한 환경이라면 그 환경에서 실행해도 됩니다. 상태가 16개뿐이므로 CPU로 충분하며 CUDA 설정은 필수가 아닙니다.

## 4. 실행 명령어

```bash
# [권장] 예제 1과 2를 50,000에피소드씩 실행
python main.py

# 1-2: MC 가치함수만 출력
python main.py --algorithm mc --episodes 50000

# 2-2: TD(0) 가치함수만 출력
python main.py --algorithm td0 --episodes 50000 --alpha 0.01

# 코드 확인용 빠른 테스트 (100에피소드, 그림 저장 생략)
python main.py --algorithm both --episodes 100 --no-plot

# 자동 테스트
python -m pytest -q
```

`--algorithm both|mc|td0`, `--seed`, `--gamma`, `--alpha`, `--episodes`, `--output-dir`, `--no-plot`, `--max-steps-per-episode`를 지원합니다. 기본값 `seed=42`를 동일하게 사용하면 두 알고리즘은 **같은 무작위 행동 궤적**을 학습하므로 결과 비교가 수월합니다.

설치한 후 `gridworld-hw3 --algorithm both` 또는 `python -m gridworld_rl.cli`로도 실행할 수 있습니다.

## 5. 제출 문항별 대응

| 과제 | 사용할 파일·결과 |
|---|---|
| 1-1 | `src/gridworld_rl/agents/every_visit_mc.py` + 공통 환경 및 실행 코드 |
| 1-2 | 터미널의 `[1-2] Every-Visit MC` 출력 또는 `results/mc_values.png` |
| 2-1 | `src/gridworld_rl/agents/td0.py` + 공통 환경 및 실행 코드 |
| 2-2 | 터미널의 `[2-2] TD(0)` 출력 또는 `results/td0_values.png` |
| 2-3 | `REPORT_ANSWERS.md`의 비교 설명 |

**스크린샷 제출에 유의:** 생성된 `.png`는 실행 결과를 그린 이미지이지 터미널 화면을 그대로 촬영한 스크린샷은 아닙니다. 교수자가 터미널 캡처를 요구한다면 `python main.py` 실행 후 `[1-2]`, `[2-2]` 결과 영역을 직접 캡처하세요.

## 6. 결과 해석

- 상태 `S15`는 종료 상태이므로 가치가 **0**입니다.
- 나머지 상태는 행동당 -1의 보상을 받으므로 가치가 **음수**입니다.
- 할인율 1에서는 어떤 상태의 가치는 무작위 정책으로 종료까지 걸리는 **예상 행동 횟수에 음수 부호를 붙인 값**입니다.
- MC·TD 값은 무작위 표본으로 추정한 수치이므로 서로 완전히 같을 필요가 없습니다.
- `[Reference]`와 `exact_values.csv`는 **학습 알고리즘의 출력이 아닙니다**. 같은 정책의 벨만 기대 방정식을 직접 풀어서 구한 정확한 비교용 수치입니다.
- TD(0)의 학습률을 고정했으므로 충분히 많은 에피소드로 학습하더라도 표본에 따른 수치 변동이 남을 수 있습니다.
