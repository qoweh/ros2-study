# PPO 핵심 정리

PPO는 Proximal Policy Optimization의 약자다. Stable-Baselines3에서 가장 자주 쓰이는 기본 강화학습 알고리즘 중 하나이고, 이 프로젝트처럼 연속 action을 가진 control 문제에 무난하게 잘 맞는다.

## PPO가 하는 일

PPO는 policy를 학습한다. policy는 observation을 받아 action을 내는 함수다.

```text
policy(obs) -> action
```

`car_rl`에서는 observation이 10차원 벡터이고 action이 2차원 벡터다.

```text
obs = [local_goal_x, local_goal_y, distance, cos_yaw_error, sin_yaw_error, ...]
action = [throttle, steering]
```

PPO는 "어떤 observation에서 어떤 action을 내야 장기 reward가 커지는가"를 반복 경험으로 배운다.

## Actor-Critic

PPO는 보통 actor-critic 구조로 이해하면 좋다.

Actor는 행동을 고른다.

```text
actor(obs) -> action distribution
```

Critic은 현재 상태가 얼마나 좋은지 추정한다.

```text
critic(obs) -> value
```

actor는 좋은 action을 더 자주 내도록 바뀌고, critic은 현재 observation에서 앞으로 받을 보상의 기대값을 더 잘 맞추도록 바뀐다.

Stable-Baselines3의 `PPO("MlpPolicy", env, ...)`에서 `MlpPolicy`는 actor와 critic이 MLP 신경망으로 만들어진다는 뜻이다.

## On-policy라는 특성

PPO는 on-policy 알고리즘이다.

on-policy는 "현재 policy로 직접 모은 데이터"를 사용해 학습한다는 뜻이다. policy가 업데이트되면 예전 policy로 모은 데이터는 오래 재사용하지 않는다.

장점은 비교적 안정적이라는 것이다. 단점은 데이터를 많이 재사용하는 off-policy 알고리즘보다 sample efficiency가 떨어질 수 있다는 것이다.

간단히 말하면 PPO는 이런 성격이다.

- 안정적이다.
- 기본값으로도 잘 돌아가는 편이다.
- 구현과 사용이 편하다.
- 데이터를 많이 먹는 편이다.
- reward 설계가 약하면 성능이 쉽게 막힌다.

## PPO의 clipping 아이디어

정책을 너무 크게 바꾸면 학습이 망가지기 쉽다. 이전 policy로 모은 데이터에 대해 새 policy가 갑자기 완전히 다른 행동을 선호하면, 업데이트가 불안정해질 수 있다.

PPO의 핵심은 policy 업데이트 폭을 제한하는 것이다. 흔히 clipped objective라고 부른다.

직관은 이렇다.

```text
좋아 보이는 방향으로 policy를 바꾸되,
한 번에 너무 멀리 바꾸지는 않는다.
```

이 특성 때문에 PPO는 이름에 `Proximal`이 들어간다. "가까운 범위 안에서 최적화한다"는 느낌으로 받아들이면 된다.

## Advantage와 GAE

PPO는 어떤 action이 평균보다 좋았는지 나빴는지를 보고 actor를 업데이트한다. 이때 쓰는 값이 advantage다.

```text
advantage = 실제로 받은 결과가 critic의 예상보다 얼마나 좋았는가
```

`gae_lambda=0.95`는 GAE, Generalized Advantage Estimation에 관련된 값이다. GAE는 advantage를 계산할 때 bias와 variance 사이를 조절한다.

대략 이렇게 이해하면 된다.

- `gae_lambda`가 낮으면 추정이 더 짧고 안정적이지만 편향될 수 있다.
- `gae_lambda`가 높으면 더 긴 미래를 반영하지만 흔들릴 수 있다.
- `0.95`는 PPO에서 자주 쓰는 무난한 값이다.

## `train.py`의 PPO 설정

현재 코드는 이렇게 PPO를 만든다.

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    tensorboard_log=str(args.log_dir / "tb"),
)
```

각 설정의 의미는 다음과 같다.

| 설정 | 의미 | 현재 값의 느낌 |
| --- | --- | --- |
| `MlpPolicy` | observation vector를 MLP로 처리 | 이미지가 아니므로 적절함 |
| `learning_rate` | 신경망 업데이트 크기 | `3e-4`는 흔한 기본값 |
| `n_steps` | 업데이트 전 모을 rollout 길이 | 1024 step씩 모아서 학습 |
| `batch_size` | update 때 나눠 먹는 mini-batch 크기 | 256이면 1024를 4묶음으로 처리 |
| `gamma` | 미래 reward 할인율 | 0.99라서 미래를 꽤 길게 본다 |
| `gae_lambda` | advantage 추정 조절 | 0.95는 무난한 PPO 설정 |
| `ent_coef` | exploration 유도 보상 | 0.01이면 행동 다양성을 조금 장려 |
| `tensorboard_log` | TensorBoard 로그 위치 | `runs/tb`에 저장 |

## PPO 학습 내부 흐름

`model.learn(total_timesteps=...)` 안에서는 대략 이런 일이 반복된다.

```text
1. 현재 policy로 env를 실행한다.
2. obs, action, reward, next_obs, done을 rollout buffer에 저장한다.
3. n_steps만큼 데이터가 쌓이면 advantage와 return을 계산한다.
4. mini-batch로 나눠 actor와 critic을 여러 번 업데이트한다.
5. 업데이트된 policy로 다시 env를 실행한다.
6. total_timesteps에 도달할 때까지 반복한다.
```

`EvalCallback`은 학습 도중 일정 간격으로 평가만 따로 수행한다. 평가 결과가 좋아지면 `runs/best_model/best_model.zip`에 저장한다.

## 이 프로젝트에서 PPO가 괜찮은 이유

`car_rl`는 PPO 입장에서 적당히 좋은 toy problem이다.

- action이 연속값이라 PPO가 자연스럽게 쓸 수 있다.
- observation이 10차원 벡터라 MLP로 충분하다.
- reward가 dense해서 초반 학습 신호가 있다.
- episode가 짧아서 시행착오가 빠르다.
- environment가 가벼워서 많은 step을 돌리기 쉽다.

다만 PPO가 모든 문제에서 최고는 아니다. 더 복잡한 환경에서는 replay buffer가 있는 off-policy 알고리즘이 sample efficiency 면에서 유리할 수 있다.

## PPO 대신 다른 알고리즘을 쓰면

### SAC

SAC는 Soft Actor-Critic이다. 연속 action control에서 매우 강한 off-policy 알고리즘이다.

특징은 다음과 같다.

- replay buffer를 사용해 과거 경험을 많이 재사용한다.
- entropy를 중요하게 봐서 탐색을 잘 유지한다.
- sample efficiency가 PPO보다 좋은 경우가 많다.
- hyperparameter와 reward scale에 민감할 수 있다.

`car_rl`에 SAC를 쓰면 같은 step 수에서 더 잘 배울 가능성이 있다. 대신 학습 설정과 로그 해석이 PPO보다 조금 더 복잡해질 수 있다.

### TD3

TD3는 Twin Delayed DDPG다. 연속 action을 위한 deterministic off-policy 알고리즘이다.

특징은 다음과 같다.

- replay buffer를 사용한다.
- deterministic policy를 쓴다.
- Q-value 과대평가를 줄이기 위해 critic 두 개를 쓴다.
- exploration noise 설정이 중요하다.

차량 제어처럼 부드러운 action이 중요한 문제에 쓸 수 있다. 다만 exploration noise가 좋지 않으면 다양한 행동을 충분히 못 해볼 수 있다.

### DDPG

DDPG는 TD3의 선배격 알고리즘이다.

특징은 다음과 같다.

- 연속 action을 다룰 수 있다.
- off-policy다.
- 현재는 TD3나 SAC가 더 안정적인 선택인 경우가 많다.

새로 실험한다면 DDPG보다는 TD3나 SAC를 먼저 보는 편이 낫다.

### DQN

DQN은 discrete action용 알고리즘이다.

`car_rl`의 action은 throttle과 steering이 연속값이므로 그대로는 DQN에 맞지 않는다. DQN을 쓰려면 action을 이산화해야 한다.

예를 들면:

```text
throttle = [-1, 0, 1]
steering = [-1, 0, 1]
총 9개 discrete action
```

이렇게 하면 구현은 단순해질 수 있지만, 조향이 거칠어지고 정밀 제어가 어려워질 수 있다.

### A2C

A2C는 Advantage Actor-Critic이다.

특징은 다음과 같다.

- PPO보다 단순하다.
- on-policy다.
- 안정성 면에서는 PPO가 더 나은 경우가 많다.
- 빠른 baseline으로는 쓸 만하다.

학습 개념을 이해하기에는 좋지만, 실제 성능 실험은 PPO가 더 편한 출발점이다.

## 선택 기준

| 알고리즘 | action 종류 | 데이터 재사용 | 장점 | 단점 |
| --- | --- | --- | --- | --- |
| PPO | discrete/continuous | 낮음 | 안정적, 기본값이 좋음 | sample efficiency가 낮을 수 있음 |
| SAC | continuous | 높음 | 탐색 좋음, 성능 좋음 | 설정이 조금 복잡함 |
| TD3 | continuous | 높음 | 부드러운 제어에 강함 | noise 설정이 중요함 |
| DQN | discrete | 높음 | 개념이 직관적 | 연속 조향에는 부적합 |
| A2C | discrete/continuous | 낮음 | 단순한 actor-critic | PPO보다 불안정할 수 있음 |

이 프로젝트의 다음 실험으로는 PPO를 조금 더 이해한 뒤 SAC를 붙여 비교하는 것이 가장 배울 게 많다.

## PPO를 공부할 때 붙잡을 핵심 문장

PPO는 현재 policy로 직접 데이터를 모으고, advantage를 계산한 뒤, policy가 한 번에 너무 크게 변하지 않도록 제한하면서 actor와 critic을 업데이트하는 알고리즘이다.

이 문장을 이해하면 `train.py`의 `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `ent_coef`가 서로 어디에 걸리는지 점점 보이기 시작한다.

