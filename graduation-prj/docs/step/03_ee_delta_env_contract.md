# EE Delta Env 계약 메모

이 문서는 현재 `option 1`로 선택한 position-only EE 제어 경로의 계약을 간단히 고정하기 위한 메모다.

## 1. 현재 선택

이번 단계에서 채택한 방향은 다음이다.

- action은 orientation 없이 `racket_center` 기준 delta xyz
- low-level sim 위에 얇은 task env를 하나 더 둠

즉 지금은 full RL wrapper가 아니라, 그 직전의 최소 계약을 만든 상태다.

## 2. 현재 action 계약

형태:
- `action.shape == (3,)`

의미:
- `(dx, dy, dz)` in meters
- 현재 target position에 더해지는 delta
- step당 절대값 `0.03m`로 clip

현재 중요한 점:
- normalization은 아직 없음
- orientation action은 아직 없음

즉 지금 action은 “실제로 얼마나 움직일지”를 바로 넣는 low-level 계약이다.

## 3. 현재 observation 계약

dict keys:
- `joint_positions`: `(7,)`
- `joint_velocities`: `(7,)`
- `racket_position`: `(3,)`
- `ball_position`: `(3,)`
- `ball_velocity`: `(3,)`

아직 빠진 것:
- racket orientation
- racket velocity
- ball spin

즉 현재 observation은 minimal version이다.

## 4. 현재 reward draft

현재는 아래 정도만 들어 있다.

- `contact_bonus`
- `height_term`
- `failure_penalty`

이건 아직 최종 shaping이 아니라, `step()` 반환 형식이 성립하는지 확인하려는 임시 초안이다.

## 5. 현재 termination 계약

termination은 기존 `failure_reason()`를 그대로 따른다.

즉 아래가 현재 종료 후보다.
- `floor_contact`
- `ball_out_of_bounds`
- `nonfinite_state`
- `ball_speed_limit`

## 6. 다음에 고정해야 할 것

다음 단계에서는 아래를 먼저 결정하는 편이 좋다.

1. observation을 dict로 유지할지 flat vector로 바꿀지
2. reward에서 height term을 유지할지, contact 중심 sparse reward로 갈지
3. termination 후 즉시 auto-reset을 env 안에서 할지, wrapper 바깥에서 할지

현재 기준으로는 아직 wrapper를 안 넣었기 때문에, reward와 observation 계약을 먼저 고정하는 것이 맞다.
