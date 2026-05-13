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

public observation은 flat vector 하나로 고정한다.

- `observation.shape == (26,)`

현재 순서는 아래와 같다.

- `observation[0:7]`: `joint_positions`
- `observation[7:14]`: `joint_velocities`
- `observation[14:17]`: `racket_position`
- `observation[17:20]`: `target_position`
- `observation[20:23]`: `ball_position`
- `observation[23:26]`: `ball_velocity`

`target_position`을 추가한 이유:
- 현재 env는 delta action을 controller 내부의 `target_position`에 누적한다.
- 이 값이 observation에 없으면 같은 `racket_position`이라도 다음 transition이 달라질 수 있다.
- 즉 RL 관점에서 state를 닫으려면 `target_position`을 같이 관측해야 한다.

디버그가 필요할 때는 다음 helper를 사용한다.

- `observation_dict()`
- `unflatten_observation()`

아직 빠진 것:
- racket orientation
- racket velocity
- ball spin

## 4. 현재 reset/step 반환 형식

이번 단계에서 Gymnasium 스타일로 반환 형식도 같이 고정한다.

- `reset() -> (observation, info)`
- `step() -> (observation, reward, terminated, truncated, info)`

현재는 time limit을 별도로 두지 않았기 때문에:

- `terminated`: `failure_reason()`이 생기면 `True`
- `truncated`: 항상 `False`

## 5. 현재 reward draft

현재는 아래 정도만 들어 있다.

- `contact_bonus`
- `height_term`
- `failure_penalty`

이건 아직 최종 shaping이 아니라, `step()` 반환 형식이 성립하는지 확인하려는 임시 초안이다.

## 6. 현재 termination 계약

termination은 기존 `failure_reason()`를 그대로 따른다.

즉 아래가 현재 종료 후보다.
- `floor_contact`
- `ball_out_of_bounds`
- `nonfinite_state`
- `ball_speed_limit`

## 7. 다음에 고정해야 할 것

다음 단계에서는 아래를 먼저 결정하는 편이 좋다.

1. reward에서 height term을 유지할지, contact 중심 sparse reward로 갈지
2. termination 후 즉시 auto-reset을 env 안에서 할지, wrapper 바깥에서 할지
3. time limit 기반 `truncated`를 추가할지

현재 기준으로는 observation/반환 형식은 고정했고, 다음은 reward와 episode 경계 정책을 정리하는 것이 맞다.
