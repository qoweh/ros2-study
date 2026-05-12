# MuJoCo 환경 구축 1차 작업 보고

## 1. 작업 목표

강화학습 전에 MuJoCo scene 수준에서 아래 요소를 하나로 묶는 것이 목표였다.

- Franka Emika Panda 로봇팔
- 탁구채
- 탁구공
- 바닥/테이블

## 2. 이번 작업에서 구현한 내용

### 2.1 자산 구성
- `~/mujoco_menagerie/franka_emika_panda`의 Franka asset을 프로젝트 내부로 복사했다.
- 프로젝트 전용 scene 파일 `pingpong_rl/assets/scene.xml`을 추가했다.

### 2.2 MuJoCo scene 구성
- floor 추가
- ping pong ball freejoint body 추가
- Franka `hand` body에 racket body 추가
- 초기 table은 제거
- 라켓 형상을 단순 원판에서 paddle 형태에 가깝게 수정
- 라켓 원점을 손잡이 그립 위치로 이동
- paddle 중심을 `racket_center` site로 분리
- 손잡이 축을 지면과 거의 평행하게 재배치
- 공 reset 위치를 `racket_center` 정중앙 위로 고정

### 2.3 Python 실행 계층 구성
- scene 로더 추가
- robot home reset 추가
- ball spawn/reset 함수 추가
- joint target 적용 함수 추가
- passive viewer 실행 스크립트 추가
- editable install용 패키징 설정 추가
- interactive viewer와 passive viewer를 분리

### 2.4 검증 코드 추가
- scene load test 추가
- ball reset test 추가
- joint ctrl buffer 반영 test 추가

## 3. 검증 결과

실행 환경:
- `conda activate mujoco_env`

검증 명령:
```bash
PYTHONPATH=pingpong_rl/src python -m unittest discover -s pingpong_rl/tests -p 'test_scene_load.py'
```

결과:
- 2개 테스트 통과
- scene XML 로딩 성공
- ball reset 및 joint target 반영 확인

## 4. 작업 중 확인한 이슈

### 4.1 include 경로와 mesh 경로 문제
- top-level scene에서 `franka/panda.xml`을 include하면 메시 경로가 어긋날 수 있었다.
- 이를 해결하기 위해 프로젝트 복사본의 `meshdir`를 scene 기준 경로로 맞췄다.

### 4.2 라켓 부착 방식 선택
- 현재는 디버깅 단순화를 위해 `hand` 자식 body 방식으로 구현했다.
- 나중에 라켓 자산을 분리하고 싶으면 weld 기반 방식과 비교가 필요하다.

### 4.3 viewer 입력 차이
- 기존 `run_viewer.py`는 `launch_passive`를 사용했다.
- 이 모드는 Python 쪽에서 직접 시뮬레이션 루프를 돌리므로 기본 MuJoCo viewer와 키 동작이 일부 다르게 느껴질 수 있다.
- 현재는 기본 실행을 interactive mode로 바꿔 MuJoCo 기본 viewer와 더 가깝게 맞췄다.

### 4.4 라켓과 공이 손 아래에 겹쳐 보이던 문제
- 원래는 racket body 원점이 paddle 쪽에 있어, 손가락이 손잡이를 잡고 있다는 시각적 해석이 깨졌다.
- 또한 공 spawn도 같은 원점을 기준으로 생각하기 쉬워 보기가 혼란스러웠다.
- 수정 후에는:
	- `racket` body 원점은 손가락 사이의 그립 위치
	- `racket_center` site는 paddle strike 중심
	- 공은 `racket_center` 위에 spawn

이 구조로 바꾸면서 집게 아래에 공이 달린 것처럼 보이는 해석 오류를 줄였다.

### 4.5 현재 reset 상태 검증
- 공 중심은 paddle 중심의 바로 위에 위치한다.
- reset 직후 contact 수는 0으로 확인했다.
- 즉 초기 상태에서 공이 로봇팔에 부딪히는 문제는 현재 기준으로 제거했다.

## 5. 다음 작업 제안

### 5.1 바로 필요한 작업
- viewer에서 라켓 면 방향 확인
- ball spawn 위치를 라켓 중심에 더 맞게 조정
- racket-ball 접촉 반발감 튜닝

### 5.2 RL 전 준비 작업
- scripted bounce baseline 작성
- EE control 계층 추가 여부 결정
- Gymnasium wrapper 도입 준비