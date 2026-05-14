# Reward와 과제 설계

강화학습에서 환경 설계의 핵심은 "무엇을 보면 되는가", "무엇을 할 수 있는가", "어떤 행동을 좋은 행동으로 볼 것인가"다. `car_rl`에서는 이 세 가지가 각각 observation, action, reward로 구현되어 있다.

## 현재 과제

차는 원점 근처에서 시작하고, 목표는 위치와 방향을 가진다.

```text
goal = [goal_x, goal_y, goal_yaw]
```

성공 조건은 두 가지를 동시에 만족하는 것이다.

- 목표 위치 가까이 들어간다.
- 목표 yaw와 현재 yaw가 충분히 비슷하다.

코드 기준은 다음과 같다.

```python
success = (
    distance < self.goal_radius
    and yaw_error < self.goal_yaw_tolerance
)
```

기본값은 다음과 같다.

- `goal_radius = 0.40`
- `goal_yaw_tolerance = 0.45 rad`

0.45 rad는 약 25.8도 정도다. 아주 정밀한 주차보다는 "목표 근처에서 대체로 방향까지 맞춘다"에 가깝다.

## Action 설계

action은 2차원이다.

```text
action[0] = throttle
action[1] = steering command
```

둘 다 `[-1, 1]`로 제한된다.

이 설계는 continuous control에 해당한다. 차를 왼쪽, 오른쪽, 앞으로 같은 discrete action으로 움직이는 것이 아니라, throttle과 steering을 연속적인 숫자로 낸다.

PPO가 이 프로젝트에 잘 맞는 이유 중 하나가 여기에 있다. Stable-Baselines3의 PPO는 `Box` 형태의 연속 action space를 자연스럽게 처리한다.

## Observation 설계

PPO는 MuJoCo 화면을 보고 학습하지 않는다. `get_obs()`가 만든 10차원 벡터만 본다.

특히 중요한 값은 다음이다.

- 목표가 차량 기준으로 앞쪽인지 옆쪽인지
- 목표까지 얼마나 남았는지
- 목표 yaw와 얼마나 틀어졌는지
- 현재 속도와 조향각이 어떤지
- 직전 action이 무엇이었는지
- episode 시간이 얼마나 남았는지

이 observation은 꽤 친절한 편이다. agent가 목표 위치를 직접 추론하지 않아도 되고, 차량 기준 local goal이 바로 들어온다. 그래서 작은 toy project에서 PPO가 학습 신호를 잡기 좋다.

## Reward 전체 구조

`step()`의 reward 계산은 다음 부분이다.

```python
progress_reward = prev_distance - distance
yaw_progress = prev_yaw_error - yaw_error
reward = (
    2.5 * progress_reward
    + 0.4 * yaw_progress
    - 0.02 * distance
    - 0.01 * float(np.square(action).sum())
)
if distance < 0.9:
    reward += 0.6 * np.cos(self._yaw_error()) - 0.03 * abs(self.speed)
if success:
    reward += 25.0
if out_of_bounds:
    reward -= 10.0
```

각 항목을 풀면 다음과 같다.

| 항목 | 의미 | 의도 |
| --- | --- | --- |
| `2.5 * progress_reward` | 이전 step보다 목표에 가까워졌는가 | 목표로 접근하게 만든다 |
| `0.4 * yaw_progress` | yaw 오차가 줄었는가 | 방향도 맞추게 만든다 |
| `-0.02 * distance` | 목표에서 멀수록 작은 벌점 | 멀리 방황하지 않게 한다 |
| `-0.01 * action^2` | action이 클수록 벌점 | 조작을 덜 거칠게 만든다 |
| `distance < 0.9`일 때 yaw bonus | 목표 근처에서 방향 정렬 보상 | 도착 직전 자세를 맞추게 한다 |
| `distance < 0.9`일 때 speed penalty | 목표 근처에서 빠르면 벌점 | 지나쳐버리지 않게 한다 |
| `success` bonus | 성공하면 큰 보상 | 진짜 목표를 분명히 알려준다 |
| `out_of_bounds` penalty | arena 밖으로 나가면 벌점 | 경계 밖으로 나가지 않게 한다 |

## Dense reward와 sparse reward

성공했을 때만 `+25`를 주는 방식은 sparse reward에 가깝다. 문제는 처음 학습하는 agent가 우연히 성공을 거의 못 하면 "뭘 잘했는지"를 배울 기회가 적어진다는 것이다.

그래서 이 환경은 dense reward를 같이 넣었다.

- 조금이라도 가까워지면 보상
- yaw 오차가 줄면 보상
- 멀리 있으면 벌점
- action을 과하게 쓰면 벌점

이렇게 하면 성공을 못 해도 "방향은 대충 맞았다", "거리는 줄였다" 같은 중간 힌트를 준다. toy project에서 매우 중요한 설계다.

## `prev_distance - distance`의 의미

`progress_reward`는 현재 거리가 아니라 거리의 변화량이다.

```text
prev_distance - distance
```

만약 이전보다 가까워졌다면 양수다. 멀어졌다면 음수다.

이 방식은 "현재 어디에 있느냐"보다 "방금 좋은 방향으로 움직였느냐"를 보상한다. 초반 학습에서 방향성을 주기 좋다.

## 목표 근처에서 reward가 달라지는 이유

멀리 있을 때와 가까이 있을 때 필요한 행동이 다르다.

멀리 있을 때는 일단 목표 근처로 가는 것이 중요하다. 가까이 왔을 때는 속도를 줄이고 yaw를 맞추는 것이 중요하다.

그래서 코드에는 이런 조건이 있다.

```python
if distance < 0.9:
    reward += 0.6 * np.cos(self._yaw_error()) - 0.03 * abs(self.speed)
```

목표 근처에서는 yaw가 잘 맞을수록 `cos(yaw_error)`가 커진다. 동시에 속도가 빠르면 벌점을 준다. 이 항목이 없으면 agent가 목표 근처를 지나치거나, 위치는 맞췄지만 방향을 못 맞추는 경우가 늘 수 있다.

## 이 reward의 한계

현재 reward는 학습용으로 괜찮지만 완벽한 주차 과제 reward는 아니다.

가능한 한계는 다음과 같다.

- 목표 근처에서 더 정밀한 감속을 배우기에는 speed penalty가 약할 수 있다.
- yaw 정렬보다 위치 접근 보상이 더 강해서, 위치만 맞추고 자세는 덜 맞출 수 있다.
- random goal 난이도가 높아지면 현재 reward만으로는 탐색이 부족할 수 있다.
- action penalty가 너무 커지면 움직이기 싫어하는 정책이 될 수 있다.
- success bonus가 너무 크면 성공 직전의 섬세한 행동보다 운 좋게 들어가는 행동을 선호할 수 있다.

## reward를 바꿀 때 보는 지표

reward를 수정할 때는 최종 reward 숫자만 보면 안 된다. 다음 값을 같이 봐야 한다.

- success rate
- mean final distance
- final yaw error
- episode length
- out_of_bounds 비율
- 목표 근처에서 속도가 얼마나 남아 있는지
- position_success는 많은데 pose_success가 적은지

`info`에 `position_success`와 `pose_success`가 이미 들어 있으므로, 다음 개선에서는 이 둘을 평가 로그로 더 잘 출력하면 좋다.

## 좋은 실험 순서

1. 현재 reward로 seed를 여러 개 바꿔 평가한다.
2. 목표 근처 speed penalty를 조금 키운다.
3. yaw alignment 보상을 목표 근처에서 조금 더 강하게 한다.
4. success 조건을 너무 어렵게 바꾸기 전에 evaluation 지표를 먼저 분리한다.
5. 학습 시간이 충분한지 확인한다.

reward는 한 번에 크게 바꾸면 원인 파악이 어려워진다. 작은 변경을 하고, 같은 명령으로 평가하는 습관이 제일 도움이 된다.

