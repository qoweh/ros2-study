# 탁구공 튕기기 강화학습 프로젝트 파이프라인 및 윤곽

본 문서는 Franka Emika Panda 로봇팔을 이용하여 탁구공을 위로 계속 튕기게 하는 강화학습(RL) 프로젝트의 전체 진행 파이프라인을 정의합니다. `RULE.md`와 `NOTICE.md`에 따라 여러 선택지와 장단점을 정리하였습니다.

---

## 1단계: 시뮬레이션 환경 구축 (Phase 1 - 현재 진행 목표)

### 1.1 Asset 로딩 기반 선택
로봇팔(Franka)과 탁구채, 공을 하나의 씬으로 구성할 때 환경을 로드/제어할 라이브러리 선택이 필요합니다.
- **Option A: 순수 MuJoCo Python 바인딩 (`mujoco`)**
  - **장점**: 직관적이고 빠름. MuJoCo 튜토리얼과 호환성이 좋음.
  - **단점**: 나중에 강화학습 환경(Gymnasium 형식)으로 감쌀 때(Wrapper) 코드를 직접 다 작성해야 함.
- **Option B: dm_control 또는 Gymnasium (`mujoco-gym`)**
  - **장점**: RL 표준 인터페이스에 맞게 구조화되어 있어 후반 작업이 편함.
  - **단점**: 커스텀 로직(공 초기화 등)을 넣을 때 프레임워크의 구조를 따라야 해서 초기 구현이 복잡할 수 있음.
> **👉 질문**: 제일 처음 만들 시뮬레이터는 **Option A(순수 MuJoCo)**로 가볍게 테스트 스크립트를 짜는 것이 디버깅에 유리합니다. Option A로 진행할까요?

### 1.2 물리 세팅 및 탁구채 부착 (Physics & Attachment)
- **라켓 부착 방식**: Franka의 end-effector(손목 부분)에 탁구채 XML을 `<body>`로 고정(`weld` 혹은 자식 노드로 통합)합니다.
- **물리/충돌 세팅 (공부 필요 📚)**: 
  - MuJoCo에서는 물체의 반발계수(bounciness)를 `solref`, `solimp` 등과 재질(Material)의 `friction` 속성으로 제어합니다. 실제 탁구공-탁구채의 반발력과 유사하도록 반복적인 튜닝이 필요합니다.

### 1.3 제어/테스트 스크립트 작성
- joint 제어가 잘 되는지, 공이 중력에 의해 떨어질 때 라켓에 맞고 튀는지 등을 3D Viewer를 통해 interactive하게 볼 수 있는 테스트 스크립트 작성.

---

## 2단계: 강화학습(RL) 파이프라인 구축 (Phase 2 - 추후 진행)

### 2.1 Task/Environment 정의 및 Action 설계
가장 큰 설계 결정은 **로봇을 어떻게 움직일 것인가(Action Space)** 입니다.
- **Option A: Joint Position/Torque Control (관절 각도/토크 직접 제어)**
  - **장점**: 모든 움직임의 가능성을 열어두어 RL에 제약을 두지 않음 (로봇 관절의 자연스러운 동역학).
  - **단점**: 탐색(Exploration) 공간이 너무 넓어 탁구채를 위로 향하게 만드는 기초 동작조차 학습이 오래 걸릴 수 있음.
- **Option B: End-Effector Operational Space Control (OSC, 데카르트 좌표 제어)**
  - **장점**: 행동 단위가 `[x, y, z 방향으로 움직여라]`가 되므로 직관적이고 학습 속도가 매우 빠름. 역운동학(IK)을 통해 관절 위치를 알아서 계산함.
  - **단점**: IK Solver 로직이 시뮬레이션 루프에 들어가야 하며, 특정 관절 특이점(Singularity)에 빠질 수 있음.
> **👉 질문**: 탁구공 튕기기는 공의 궤적이 중요하므로 X, Y, Z 방향으로 제어하는 **Option B (End-Effector 위치 제어)**가 성공 확률이 높습니다. 어떤 방식이 끌리시나요?

### 2.2 Observation 설계
- 로봇팔 상태: 각 관절(또는 EE)의 위치 및 속도.
- 물체 상태: 탁구공의 3D 위치(x, y, z) 및 속도(vx, vy, vz). (TODO.md에 따라 카메라 없이 공의 위치는 완벽히 안다고 가정).

### 2.3 Reward 설계 (Reward Shaping)
- **Dense Reward**: 
  - 공이 라켓 중앙(sweet spot)에 가깝게 있을수록 플러스(+)
  - 공의 높이(z)가 일정 높이를 유지하고, z 방향 속력이 적절할 때 플러스(+)
- **공부 필요 📚**: 연속 제어에서 Reward 구조가 잘못되면 로봇이 가만히 있거나 이상하게 움직이는 꼼수(Reward Hacking)를 배우게 됩니다. (예: 공을 튕기지 않고 라켓 위에 가만히 얹어두기).

### 2.4 알고리즘 및 프레임워크 선택
- **Framework**: Stable-Baselines3 (SB3) vs CleanRL. (SB3가 M1 Mac(`mps`) 대응 및 초보자 접근성에 유리합니다).
- **Algorithm**: PPO (안정적) vs SAC (효율적). 

---

## 다음 진행을 위한 선택 요청

위 윤곽을 살펴보시고 다음 항목을 결정해주시면, **1단계 시뮬레이션 환경 구축** 코드 작성을 바로 시작안내하겠습니다.

1. **Asset & 기초 생성 옵션**: 순수 `mujoco` Python 바인딩(Option A)으로 인터랙티브 뷰어가 포함된 기본 환경 구축부터 시작하는 것에 동의하시나요?
2. 이후 강화학습을 고려하여, 폴더 구조를 일반적인 Python 패키지(예: `src/` 내 분리)로 만들고 진행할까요? 
3. 기타 수정하고 싶은 단계가 있으신가요?# 탁구공 튕기기 강화학습 프로젝트 파이프라인
본 프로젝트는 Franka Emika Panda 로봇팔을 사용하여 탁구공을 지속적으로 위로 튕기는 강화학습(RL) 환경을 구축하는 것을 목표로 한다.  
현재 목표는 **강화학습 이전 단계인 MuJoCo 기반 시뮬레이션 환경 안정화**이다.
---
# 1단계: MuJoCo 시뮬레이션 환경 구축 (현재 목표)
## 1.1 초기 구현 방향
초기 구현은 **순수 MuJoCo Python 바인딩 (`mujoco`) 기반**으로 진행한다.
### 선택 이유
- Viewer 기반 디버깅이 쉬움
- collision/contact 튜닝에 유리
- physics 문제를 low-level에서 직접 확인 가능
- RL wrapper(Gymnasium)는 이후 추가 가능
- 초기 목표는 RL보다 physics/control 안정화에 있음
초기에는 다음과 같은 구조로 interactive 테스트를 수행한다.
```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

⸻

## 1.2 프로젝트 구조

초기부터 Python 패키지 구조로 관리한다.
```
(파일이름 및 디렉토리 이름과 구조 참고 - 똑같이 안 해도 됨.)
pingpong_rl/
├── assets/
│   ├── franka/
│   ├── racket/
│   ├── ball/
│   └── scene.xml
│
├── src/
│   ├── envs/
│   │   ├── base_env.py
│   │   └── pingpong_env.py
│   │
│   ├── controllers/
│   │   ├── joint_controller.py
│   │   └── ee_controller.py
│   │
│   ├── physics/
│   │   ├── contacts.py
│   │   └── randomization.py
│   │
│   ├── utils/
│   │   ├── viewer.py
│   │   └── reset.py
│   │
│   └── train/
│       ├── train_ppo.py
│       └── eval.py
│
├── tests/
├── requirements.txt
└── README.md
```
⸻

## 1.3 Scene 구성

MuJoCo scene에는 다음 요소를 포함한다.

구성 요소

* Franka Emika Panda 로봇팔
* 탁구채 (racket)
* 탁구공 (ping pong ball)
* 테이블 또는 기준 plane

⸻

## 1.4 탁구채 부착 방식

탁구채는 Franka end-effector에 고정한다.

방식

* XML <body> 자식 노드로 부착
* 또는 weld constraint 사용

예시 개념:
```
<body name="racket" pos="0 0 0.1">
    ...
</body>
```

⸻

## 1.5 Physics 세팅

탁구는 contact physics 품질이 매우 중요하다.

핵심 요소

* timestep
* contact solver
* restitution 느낌의 bounce tuning
* friction
* penetration 방지

초기 추천값(참고):
```
<option timestep="0.002"/>
<geom
    solref="0.002 1"
    solimp="0.95 0.99 0.001"
/>
```
※ 실제 탁구 느낌을 위해 반복 튜닝 필요.

⸻

## 1.6 Control Frequency 설계

Physics timestep과 RL action frequency를 분리한다.

추천 구조

Physics timestep : 500Hz (0.002 sec)
Control timestep : 50Hz  (0.02 sec)

이유

* physics는 충돌 안정성을 위해 고주파 필요
* RL action은 너무 빠르면 학습이 불안정해짐

⸻

## 1.7 초기 테스트 목표

강화학습 이전에 다음이 안정적으로 동작해야 한다.

체크리스트

* 로봇 joint control 가능
* EE(end-effector) 이동 가능
* 공 spawn/reset 가능
* 공이 중력에 따라 자연스럽게 떨어짐
* 라켓 충돌 시 정상적으로 튐
* viewer에서 interactive 확인 가능

⸻

# 2단계: 강화학습 환경 구축

⸻

## 2.1 Action Space 설계

추천 방식

End-Effector Position Control (Operational Space Control)

예시:

action = [dx, dy, dz]

또는:

action = [x, y, z, yaw]

⸻

EE Control 선택 이유

장점

* 학습 난이도 감소
* 탐색 공간 축소
* 공 trajectory tracking에 적합
* 직관적인 action space

단점

* IK(역기구학) 필요
* singularity 문제 가능성 존재

⸻

비추천 (초기 단계)

Joint torque/action 직접 제어

이유:

* exploration 난이도 매우 높음
* 학습 수렴 속도 느림
* 초기 탁구 task에는 비효율적

⸻

## 2.2 Observation 설계

로봇 상태

* EE position/velocity
* 또는 joint position/velocity

공 상태

* ball position (x, y, z)
* ball velocity (vx, vy, vz)

초기에는 완벽한 state access 허용.

⸻

## 2.3 Reward 설계

초기 reward는 단순하게 시작한다.

추천 초기 reward

+1 if ball contacts racket

⸻

이후 추가 가능 요소

* 공 높이 유지
* 중앙 타격(sweet spot)
* 안정적인 trajectory
* energy penalty
* action smoothness

⸻

주의점

Reward hacking 가능성 존재.

예:

* 공을 튕기지 않고 라켓 위에 올려두는 행동 학습

따라서 reward shaping은 단계적으로 진행.

⸻

## 2.4 Scripted Policy 선행 권장

RL 전에 heuristic controller를 먼저 만든다.

예:

* 공 아래로 이동
* 타이밍 맞춰 위로 치기

⸻

이유

Scripted baseline이 있으면:

* physics 문제 검증 가능
* observation 검증 가능
* reward 문제 검증 가능

Robotics RL에서는 매우 중요함.

⸻

2.5 RL 프레임워크

추천

* Stable-Baselines3 (SB3)

이유

* 사용 쉬움
* 자료 많음
* PPO 지원 안정적
* Apple Silicon(MPS) 환경 대응 편함

⸻

## 2.6 추천 알고리즘

초기 추천

PPO

이유

* 안정적
* continuous control에서 많이 사용됨
* robotics baseline으로 적합

⸻

# 3단계: 추천 개발 순서

STEP 1

Franka asset 로딩

STEP 2

라켓 attach

STEP 3

탁구공 freejoint 추가

STEP 4

gravity/drop 테스트

STEP 5

contact physics 튜닝

STEP 6

EE controller 구현

STEP 7

scripted bouncing 구현

STEP 8

Gymnasium wrapper 추가

STEP 9

PPO 학습 시작

⸻

핵심 목표 요약

현재 가장 중요한 것은:

1. Physics 안정화
2. Control abstraction 설계
3. Reset robustness 확보

강화학습 자체보다도:

* collision 품질
* timestep 안정성
* contact tuning
* reset consistency

가 프로젝트 성공에 더 중요하다.
