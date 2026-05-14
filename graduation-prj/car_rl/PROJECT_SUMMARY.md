# car_rl 심화 토이프로젝트 정리

## 무엇을 바꿨나
기존 `car_rl`은 `x` 축 슬라이더 하나만 제어하는 1차원 목표 도달 예제였다. 이번 작업에서는 같은 디렉토리 안에서 다음처럼 확장했다.

- `x` 축만 있던 상태를 `x, y, yaw`를 가지는 2D 차량 과제로 확장했다.
- 액션을 `1개 motor`에서 `throttle, steering` 2개로 바꿨다.
- 목표를 고정 `x=2.0` 하나에서 `랜덤 목표 위치 + 목표 yaw`로 바꿨다.
- MuJoCo 씬에 자동차 바디와 4개 wheel 시각 요소를 추가했다.
- PPO 학습 스크립트, 평가 스크립트, 스크립트형 데모 진입점을 새 과제에 맞게 다시 만들었다.

## 현재 과제 정의
핵심 구현은 `car_env.py`에 있다.

- 상태 공간: 총 10차원
- 포함 정보:
  - 차량 기준 local goal x, y
  - goal까지 거리
  - goal yaw와의 오차를 `cos`, `sin`으로 표현한 값
  - 현재 속도
  - 현재 steering 각도
  - 직전 액션
  - 남은 episode 비율
- 액션 공간: `[-1, 1]` 범위의 2차원
  - `action[0]`: throttle
  - `action[1]`: steering command
- 종료 성공 조건:
  - goal distance `< 0.40`
  - yaw error `< 0.45 rad`
- episode 길이:
  - 기본 `240 step`

## 목표 샘플링 방식
완전 랜덤 yaw는 짧은 PPO 학습에서 너무 어려워서, 랜덤 목표 위치는 유지하되 목표 yaw는 대체로 그 목표 방향을 따르도록 샘플링했다.

즉:

- 목표 위치는 랜덤
- 목표 yaw는 목표 위치 방향을 중심으로 약간의 노이즈를 더해 샘플링
- 사용자가 수동 goal을 줄 때는 원하는 yaw를 그대로 넣을 수 있음

이렇게 해서 `회전`은 남기되, PPO가 실제로 학습 신호를 잡을 수 있게 만들었다.

## 파일별 역할
- `car_env.py`
  - 2D 차량 환경, reward, goal sampling, success 조건
- `car.xml`
  - chassis, cabin, front/rear wheel, goal marker가 들어간 MuJoCo 씬
- `train.py`
  - PPO 학습 진입점
  - EvalCallback 포함
  - 기본 출력 모델: `car_ppo_nav.zip`
- `test.py`
  - 학습된 모델 평가
  - 랜덤 goal 또는 수동 goal 평가 가능
- `main.py`
  - 학습 없이 scripted controller로 환경을 바로 확인하는 데모

## 실행 방법
작업 디렉토리:

```bash
cd car_rl
```

학습:

```bash
python train.py --timesteps 50000 --eval-freq 10000 --eval-episodes 5
```

기본 평가:

```bash
python test.py --episodes 15 --headless
```

렌더링 포함 평가:

```bash
python test.py
```

수동 goal 평가 예시:

```bash
python test.py --goal-x 1.8 --goal-y -1.2 --goal-yaw 1.57
```

학습 없이 scripted demo 보기:

```bash
python main.py --goal-x 1.8 --goal-y -1.2 --goal-yaw 1.57
```

## 이번 작업에서 실제 확인한 결과
기본 모델 경로(`car_ppo_nav.zip`)로 5만 step 학습 후, 다음 평가를 실행했다.

```bash
python test.py --episodes 15 --headless
```

관찰 결과:

- 성공률: `0.40`
- 평균 보상: `53.333`
- 평균 최종 거리: `0.637`
- 성공한 episode들은 대체로 `29 ~ 36 step` 안에서 빠르게 goal에 들어갔다.

즉, 단순히 앞으로 밀기만 하는 예제가 아니라, 적어도 일부 랜덤 목표에 대해 `방향을 잡고 접근해서 자세까지 어느 정도 맞추는 정책`이 학습된 것은 확인했다.

## 해석
현재 baseline은 다음 성격을 가진다.

- 가까운 goal에는 꽤 빠르게 수렴한다.
- 일부 실패 episode는 goal 근처까지는 가지만 yaw 허용 오차를 끝까지 못 맞추고 시간을 다 쓴다.
- 그래서 다음 개선 포인트는 `정밀 자세 정렬` 구간을 더 잘 학습시키는 것이다.

## 바로 다음 개선 후보
다음 실험은 우선순위가 높다.

1. goal 근처에 들어오면 속도를 더 강하게 줄이도록 reward를 추가
2. 성공 조건을 `position_success`와 `pose_success`로 나눠 로그 저장
3. obstacle을 1~2개 추가해서 단순 접근이 아니라 경로 선택까지 학습
4. `train.py`에 학습 곡선 저장용 CSV 혹은 별도 summary 출력 추가
5. random start pose 범위를 조금 더 넓혀 일반화 확인
