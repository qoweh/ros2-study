# Ball Bounce와 EE Delta Env 도움말

이 문서는 이번 단계에서 추가된 두 가지를 설명한다.

- 왜 ball이 이전에는 거의 튀지 않았는가
- `EE delta xyz` 기반 env를 왜 지금 형태로 만들었는가

## 1. 왜 이전 ball은 거의 안 튀었는가

문제는 크기나 질량보다 contact 응답이었다.

현재 프로젝트에서 ball 스펙 자체는 이미 꽤 맞았다.

- radius: `0.02m`
- mass: `0.0027kg`

즉 외형과 질량은 탁구공에 가까웠다.

하지만 floor drop을 해보면, 첫 충돌 뒤 최고점이 거의 바닥 높이 근처에 머물렀다.

원인:
- ball `solref`가 너무 damping-heavy
- floor도 기본 plane 상태라 rebound 성향이 약함
- 두 geom의 조합 결과가 “가벼운 공”이 아니라 “잘 안 튀는 공”처럼 보였음

## 2. 이번에 어떤 식으로 조정했는가

이번 단계에서는 완전 현실적인 탁구 physics를 바로 넣지 않았다.

대신 목표를 이렇게 잡았다.

- floor에 닿으면 눈에 띄는 rebound가 있어야 함
- 그래도 지나치게 chaotic한 high-bounce는 피해야 함

그래서 pure floor drop 기준으로 여러 조합을 직접 비교했고, 그중 moderate bounce에 해당하는 값을 선택했다.

현재 선택:
- ball
  - `friction="0.02 0.001 0.0001"`
  - `solref="0.001 0.08"`
- floor
  - `friction="0.2 0.005 0.0001"`
  - `solref="0.0006 0.03"`

이 값으로 1m floor drop 기준 post-impact peak가 약 `0.371m`까지 올라온다.

즉 완전 현실적인 탁구공은 아니지만, 이전처럼 바닥에 붙는 수준은 벗어났다.

## 3. 왜 여기서 멈췄는가

더 강하게 튀게 만들 수도 있었다.

예를 들어 더 낮은 damping 조합은 1m drop에서 `0.5m` 이상 또는 그 이상으로도 튀었다.

하지만 지금 단계에서 그 값을 바로 쓰지 않은 이유는 다음과 같다.

- 제어기 디버깅이 더 어려워진다.
- reward/termination이 아직 고정되지 않았다.
- orientation도 아직 안 들어간 상태라 원인 분리가 어려워진다.

즉 이번 선택은:
- realism 최우선
n이 아니라
- control architecture 안정화 우선

이라는 현재 프로젝트 방향에 맞춘 타협값이다.

## 4. EE delta env는 왜 새 파일로 분리했는가

선택지는 두 가지였다.

- 기존 `PingPongSim`에 RL용 action/reward/termination까지 전부 섞기
- low-level sim은 유지하고, 그 위에 얇은 task env를 하나 더 두기

이번에는 두 번째를 선택했다.

이유:
- `PingPongSim`은 scene/reset/contact 같은 low-level 역할만 유지할 수 있다.
- RL 인터페이스 실험을 해도 기존 시뮬레이터를 덜 건드린다.
- observation/action/reward를 이후에 바꾸기 쉽다.

현재 새 env는 `PingPongEEDeltaEnv`다.

## 5. 현재 action 계약

현재 action은 orientation 없이 위치 3축만 쓴다.

형태:
- `(dx, dy, dz)`

현재 의미:
- `racket_center` target 위치에 더할 delta position
- 1 step당 `action_limit=0.03m`로 clip

즉 지금은 “라켓 중심점을 얼마나 이동시키고 싶은가”만 다룬다.

## 6. 현재 observation 계약

현재 observation은 dict 형태로 고정했다.

키:
- `joint_positions`
- `joint_velocities`
- `racket_position`
- `ball_position`
- `ball_velocity`

이건 아직 Gymnasium 최종 형식은 아니고, 다음 단계에서 flatten할지 dict 유지할지 결정할 수 있는 중간 계약이다.

## 7. 현재 reward/termination은 어떤 수준인가

reward는 아직 draft다.

현재 항목:
- racket contact bonus
- ball이 racket보다 위에 있을 때의 작은 height term
- floor/failure penalty

termination은 현재 `failure_reason()`에 맞춘다.

즉 지금 단계의 목적은 “학습이 잘 되게 reward를 완성”이 아니라, `env.step(action)` 형태가 실제로 성립하는지 고정하는 것이다.
