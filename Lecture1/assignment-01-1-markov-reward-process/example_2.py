import torch

from markov_reward_process import MarkovRewardProcess

def main():

    states = [
        "Class1",
        "Class2",
        "Class3",
        "Pass",
        "Pub",
        "Facebook",
        "Sleep"
    ]

    state_transition_matrix = torch.tensor(
        [
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]
    )

    rewards = torch.tensor(
        [
            -2,
            -2,
            -2,
            10,
            1,
            -1,
            0
        ]
    )

    discount_factor = 0.9





    mrp = MarkovRewardProcess(
        states=states,
        transition_matrix=state_transition_matrix,
        rewards=rewards,
        gamma=discount_factor,
        seed=42,
    )





    trajectory = mrp.simulate(
        start_state="Class1",
        num_steps=10,
    )

    print("Trajectory")
    print("-" * 60)

    for transition in trajectory:
        print(
            f"Step {transition['step']:2d}: "
            f"{transition['state']} "
            f"-> {transition['next_state']} "
            f"| Reward = {transition['reward'].item():.2f}"
        )





    total_return = mrp.discounted_return(
        trajectory
    )

    print()
    print(
        f"Discounted return: "
        f"{total_return.item():.6f}"
    )





    iterative_values = (
        mrp.value_function_iterative()
    )

    exact_values = (
        mrp.value_function_exact()
    )





    print()
    print("Value Function")
    print("-" * 60)

    print(
        f"{'State':10s}"
        f"{'Iterative':>15s}"
        f"{'Exact':>15s}"
    )

    for state, iterative_value, exact_value in zip(
        states,
        iterative_values,
        exact_values,
    ):
        print(
            f"{state:10s}"
            f"{iterative_value.item():15.6f}"
            f"{exact_value.item():15.6f}"
        )


# python main.py를 실행할 경우 __name__ == "__main__" 이 된다.
# 따라서 main() 함수가 실행된다.
# 그러나, 다른 파일에서 import main 코드를 넣으면
# 그 파일이 실행될 때 __name__ == "main" 이 된다.

# 다른 파일에서 import main 코드를 넣으면
# 1. main.py 파일을 찾음
# 2. main.py의 코드를 위에서 아래로 실행함
# 3. 함수와 클래스 정의를 생성함
# 4. 최상위에 있는 일반 명령문도 실행함
# 5. 만들어진 모듈 객체를 import한 곳에 연결함

# python a.py를 실행해도 __name__ == "__main__" 이 된다.
# __name__ == "__main__" 은 프로그램의 시작점으로 실행된
# 모듈에 특별히 부여되는 것이기 때문이다.
if __name__ == "__main__":
    main()
