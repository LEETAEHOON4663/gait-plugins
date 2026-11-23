# gait-plugins

AI 수업 팀 프로젝트의 가능성 시연입니다. 스켈레톤 하나를 공통 입력으로 받아, 질병별 연구 모델이 각자 필요한 정보만 가져가 판정하는 플러그인 구조입니다. 연구용 시연이며 진단 도구가 아닙니다.

```
NVIDIA 77관절 JSONL
   → core/nova77.py    형식 검사, 공통 스켈레톤(77관절 전체, 읽기 전용)
   → core/common.py    5프레임 이하 끊김 보간(표시 남김), 보행 구간, 촬영 방향, 보행 축
   → core/view.py      바닥 기준 수직 추정, (앞, 왼쪽, 위) 3D, 연구별 가상 카메라
   → core/plugin.py    플러그인마다 조건 검사 → 실행 → 결과 (하나가 실패해도 나머지는 계속)
        ├─ plugins/pd_stgcn_sabo2022         공개 가중치 그대로 (정면, 파킨슨 보행 점수, 3D)
        ├─ plugins/pd_stgcn_sabo2022_ap2d    같은 연구의 2D 모델 (AlphaPose용)
        ├─ plugins/pd_stgcn_sabo2022_dt2d    같은 연구의 2D 모델 (Detectron용)
        ├─ plugins/gait_gaitlab2020          공개 가중치 그대로 (측면, 속도·걸음수·무릎·GDI)
        └─ plugins/gait_events_zeni2008      공개 계산법 (걸음 시점)
```

각 연구는 `Camera(azimuth, elevation, output)`만 선언하면, 어느 방향에서 찍은 영상이든 그 연구의 카메라에서 본 3D 또는 2D 좌표를 받습니다. 예: ST-GCN은 `Camera(azimuth=0, elevation=30)`(걷는 사람 정면, 30도 아래로).

공통층은 값을 버리지 않습니다. 중심 이동, 크기 조정, 자르기 같은 연구별 정규화는 각 플러그인의 `adapt` 안에만 있습니다.

## 바로 시작

Python 3.10에서 시험했습니다(3.9~3.12). GPU는 없어도 됩니다. Windows에서는 WSL에서 실행하세요.

```bash
git clone https://github.com/LEETAEHOON4663/gait-plugins.git
cd gait-plugins
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/setup.py
python -m pytest -q
```

- `pip install`은 torch 때문에 몇 GB를 받습니다. GPU가 없으면 먼저 `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu`를 실행하면 훨씬 작습니다.
- `scripts/setup.py`는 두 연구의 공개 저장소를 시험한 커밋 그대로 `third_party/`에 받고(약 330MB), 공개 가중치를 각 플러그인의 `weights/`에 넣습니다. 가중치는 연구의 라이선스를 따르므로 이 저장소에는 올리지 않습니다.
- 테스트는 68개가 모두 통과해야 합니다.

**예시 파일로 실행** (데이터 없이 바로, [samples/](samples/README.md))

```bash
python scripts/run.py samples/004_NM_01.jsonl.gz samples/001_PD_01_MD.jsonl.gz --size 1280x720
python scripts/run.py samples/HOA001_M0_gait03.60948372.jsonl.gz --size 644x484
```

**이식한 모델이 원본과 같은지 확인** (연구 저장소의 예시 데이터 사용)

```bash
python scripts/verify_stgcn_port.py third_party/stgcn_parkinsonism_prediction
python scripts/verify_sabo2d_port.py third_party/stgcn_parkinsonism_prediction
python scripts/verify_gaitlab_port.py third_party/mobile-gaitlab/demo plugins/gait_gaitlab2020/weights/gaitlab.npz
```

**시연 화면**: `demo/index.html`을 브라우저로 열면 됩니다(설치 불필요).

## README 수치 다시 만들기

NVIDIA 추출 전체(760개, 압축 해제 시 3.6GB)가 필요합니다. 팀 Drive의 Nova-77 추출본에서 `frame/`과 `frame_reports/` 두 폴더를 `data/nova77_full/` 아래에 두세요(`data/nova77_full/frame/KOA-PD-NM/...`, `data/nova77_full/frame_reports/batch_status_v0_1.jsonl`이 보이면 됩니다).

| 수치 | 명령 | 저장소의 결과 (화면 출력이 이 로그와 같아야 함) |
| --- | --- | --- |
| ST-GCN AUC 표 (KOA-PD-NM) | `python scripts/evaluate.py data/nova77_full results` | `results/evaluate.log` |
| 2D 모델·걸음 시점 규칙 | `python scripts/evaluate_new.py data/nova77_full results` | `results/new_plugins.log` |
| mobile-gaitlab 걸음수 비교 | `python scripts/evaluate_gaitlab.py data/nova77_full results` | `results/gaitlab.log` |
| 고관절 정면, 수술 전후 | `python scripts/evaluate_hip_front.py data/nova77_full results` | `results/hip_front.log` |
| 측면 걷는 방향 차이 | `python scripts/side_direction_check.py data/nova77_full results` | `results/side_direction.log` |
| 무릎 자체 모델 (예제) | `python examples/koa_gait_baseline/train.py data/nova77_full results` | `results/train.log` |
| 시점 변환 표 | `python scripts/compare_views.py data/nova77_full` | `results/compare_views.log` |
| 시연 화면 다시 만들기 | `python scripts/export_demo.py data/nova77_full results/demo.json` 후 `python scripts/build_demo.py` | `demo/index.html` |

TOAGA 관련 수치는 공개 데이터라 바로 받을 수 있습니다(영상 제외 약 70MB).

```bash
python scripts/fetch_toaga.py data/toaga
python scripts/validate_zeni_on_toaga.py data/toaga/x/Xsens/Xsens
python scripts/toaga_baseline.py data/toaga/x/PoseTracking third_party/stgcn_parkinsonism_prediction results
python scripts/toaga_selftest.py data/toaga/x/PoseTracking third_party/stgcn_parkinsonism_prediction /tmp/toaga_selftest
```

- 위 표의 평가 스크립트는 2개 프로세스로 파일을 하나씩 읽습니다. 공용 서버에서는 `nice -n 19`를 붙여 주세요.
- `check_world.py`, `check_fidelity.py`, `check_front_view.py`, `compare_views.py`, `explore_*.py`는 탐색용입니다. 6개 프로세스로 고관절 571개를 한꺼번에 메모리에 올리므로 공용 서버에서는 돌리지 마세요. `results/fidelity.log`, `results/world.log`는 화면 밖 구간을 자르기 전 기준입니다.

## 파일 구성

| 경로 | 내용 |
| --- | --- |
| `core/` | 입력 읽기, 공통 정리, 3D 시점 변환, 플러그인 실행 |
| `plugins/` | 연구 플러그인 5개 (폴더를 지우면 빠지고, 넣으면 추가됨) |
| `shared/` | 여러 플러그인이 같이 쓰는 모델 코드 (ST-GCN, 걸음 지표) |
| `scripts/` | 실행, 원본 대조, 평가, TOAGA 판정 |
| `samples/` | 예시 기록 3개 |
| `results/` | 이 README의 수치를 만든 출력 |
| `requests/` | TOAGA 영상 NVIDIA 추출 요청서 |

**연구를 더 추가하며 알게 된 것**

- 같은 연구의 2D 모델 두 개는 서로 잘 맞습니다(피험자 단위 순위 상관 0.93). 3D 모델과는 덜 맞습니다(0.64~0.71). 같은 연구라도 **모델을 바꾸면 결과가 달라집니다.**
- AlphaPose 2D 모델은 우리 입력에서 정상군도 평균 1.16점으로 높게 나와 절대값은 쓸 수 없습니다(PD 1.53, NM 1.16). 순서와 AUC(PD 대 NM: 3D 0.98, AlphaPose 0.94, Detectron 0.94)는 유지됩니다.
- 정면 고관절(멀어지는 방향, 온몸이 화면 안인 구간만): 같은 세션 반복 차이는 3D 0.17, AlphaPose 2D 0.07, Detectron 2D 0.12점. 수술 전후 변화는 1.05→0.81, 1.50→1.45, 1.07→0.98이고, 점수가 내려간 사람은 각각 38/50, 29/54, 30/54명입니다. **3D 모델만 일관되게 내려가고 2D 모델은 거의 우연 수준**이라 '수술 효과'라고 말할 수 없습니다.
- 몸 크기: 연구 2D 모델은 크기를 정규화하지 않으므로, 연구 학습 데이터의 몸 높이(291px)에 맞춰 키웁니다. 카메라 기울기는 0으로 가정했습니다.

**검토했지만 넣지 않은 연구**

- **걸음 감정 (ProxEmo, STEP):** 가중치와 시험 데이터가 있지만, **감정 이름의 순서가 저장소마다 다릅니다.** 저자 시연 코드는 0=화남·1=행복인데, 같은 데이터에서 화남으로 연기한 걸음의 라벨은 1이 11개·0이 3개로 반대입니다. 뜻을 확신할 수 없어 넣지 않았습니다. 환자에게 감정 이름을 붙이는 것도 조심스럽습니다.
- **무릎 하중 (predictKAM):** 가중치는 있지만 골반 표식(ASIS·PSIS)과 발뒤꿈치가 필요합니다. 77관절에 없어 추정으로 채워야 해서 보류했습니다.
- **OpenCap 마커 보강:** 발뒤꿈치·새끼발가락이 필요합니다.
- **동결보행(MS-GCN):** 가중치가 공개돼 있지 않습니다.

## 시연 화면

`demo/index.html`을 브라우저로 열면 됩니다(데이터가 들어 있어 서버가 필요 없음). 예시 5개가 파이프라인을 통과하는 모습, 3D 회전, 연구마다 받는 관절과 결과, 거절 사유를 보여 줍니다. 값은 모두 실제 파이프라인 출력입니다. 관절 좌표는 CC BY 4.0 공개 데이터셋에서 추출했습니다(아래 출처).

## 플러그인 추가

`plugins/<이름>/plugin.py`에 `Plugin`을 상속한 클래스를 만들고 `PLUGIN`에 넣으면 됩니다. core는 고치지 않습니다. 빼려면 폴더를 지웁니다.

- `manifest.requires`: 필요한 관절 세트, 촬영 방향, 최소 길이, 관측 비율
- `adapt(p)`: 필요한 관절만 꺼내 연구의 정규화 적용
- `predict(x)`: 연구 모델 실행, `label`과 `score` 반환
- `manifest.fidelity`: 성능이 유지된다는 근거. 결과마다 함께 표시됩니다

조건이 안 맞으면 점수 대신 사유와 함께 "판단 불가"가 나옵니다.

## 지금 들어 있는 연구

| 플러그인 | 무엇을 탐지하나 | 원본과 같은가 | 우리 데이터에서 |
| --- | --- | --- | --- |
| `pd_stgcn_sabo2022` | 파킨슨 보행 점수(UPDRS-gait 0~2) | 코드 같음 (차이 3e-7) | 정답이 없어 정확도 미확인 |
| `gait_gaitlab2020` | 속도·걸음수·무릎 굴곡·GDI (질병 아님) | 코드 같음, 저자 예시 영상의 7개 값과 일치 (차이 2.5e-6) | 걸음수를 직접 센 값과 순위 상관 0.69~0.70. 절대값은 +0.09 걸음주기/초(약 12%) 높음 |
| `pd_stgcn_sabo2022_ap2d`, `pd_stgcn_sabo2022_dt2d` | 같은 연구의 2D 모델 (AlphaPose용, Detectron용), 정면 | 코드 같음 (저자 샘플 10개 모델 × 8걸음 차이 0.0) | 아래 참고. 절대값이 아니라 순서만 |
| `gait_events_zeni2008` | 걸음수·보행주기·디딤 비율 (공개 계산법, 가중치 없음) | 계산법 그대로 (발뒤꿈치 대신 발목) | TOAGA 모션캡처 센서(정답) 11명에 같은 규칙을 적용하면 걸음수 104.9 (표준편차 8.8), 논문 발표값 105.2 (9.1) |
| `examples/koa_gait_baseline` | 무릎 관절염 점수 | 발표된 연구 아님 (자체 기준) | 활성 플러그인에서 제외하고 예제로 보관 |

**측면 연구 mobile-gaitlab** (Kidzinski 2020): 측면 2D를 25프레임/초가 아니라 **30프레임/초 그대로** 넣습니다. 저자 예시 영상이 29.97fps인데 저자 코드가 그대로 사용하고, 코드의 `FPS = 25` 상수는 쓰이지 않습니다. 처음에 25로 변환했다가 걸음수가 30% 치우쳐서 발견했습니다. 최소 126프레임(약 4.2초)이 필요해 짧게 걷는 정상군 대부분은 제외됩니다(109/189개 실행, 정상 2명뿐). 걸음수의 평균 오차(0.110)는 평균값만 찍어도 나오는 수준(0.108)이라 **절대값은 믿을 수 없고 순위만 참고**합니다. 학습 데이터가 뇌성마비 아동 위주라 성인에게는 어긋납니다. GDI는 같은 사람 두 영상에서 평균 8점 차이가 납니다. 검증: `scripts/verify_gaitlab_port.py`, `scripts/evaluate_gaitlab.py`.

## 데이터

KOA-PD-NM 189개(KOA 100, NM 59, PD 30)와 Hip_OA 571개. 2개는 추출 실패(원본이 빈 파일 1개, 시간 초과 1개). 760개 모두 `bbox`가 없고 한 사람만 나오며 30fps입니다.

## 결과 (KOA-PD-NM, 피험자 단위)

**파킨슨 ST-GCN, 재학습 없이 그대로 적용 (NM 30 / PD 16 / KOA 50명)**

| 입력 처리 | PD vs NM AUC | PD vs KOA AUC |
| --- | --- | --- |
| 연구 방식대로 3D 변환 | 0.98 | 0.80 |
| 카메라 기울기 변환 생략 | 0.91 | 0.74 |
| 방향 변환 생략 | 0.77 | 0.65 |
| (비교) 걷는 속도만 사용 | 1.00 | 0.86 |

- 연구별 변환을 생략하면 성능이 크게 떨어집니다. 플러그인마다 변환을 따로 두는 이유입니다.
- 높은 AUC는 파킨슨을 알아본다는 증거가 아닙니다. 이 데이터에서는 걷는 속도만으로도 더 높은 값이 나옵니다. PD와 KOA 구분에서 속도에 점수를 더해도 0.848에서 0.861로 거의 늘지 않습니다.
- 바닥 보정을 넣은 뒤 PD vs KOA AUC는 0.85에서 0.80으로 내려갔습니다(PD 16명). 보정은 라벨이 아니라 같은 사람의 일치도를 기준으로 골랐습니다.
- 무릎 환자 50명 중 43명이 UPDRS 1점 이상으로 나옵니다. 파킨슨 전용이 아니라 전반적인 보행 이상에 반응합니다.
- 고관절 정면 자료의 절반 가까이는 걷는 프레임에 화면 밖 관절이 섞여 있었습니다. 온몸이 보이는 구간만 쓰니 프레임의 33%가 빠지고, 수술 전후 변화는 1.05→0.81(3D, 50명 중 38명 하락)로 작아졌습니다(보정 전 1.16→0.86).

**무릎 관절염, 새로 학습 (KOA 50 / NM 30 / PD 15명, 중첩 교차검증)**

| 비교 | AUC |
| --- | --- |
| 전체 (정상 + 파킨슨 대비) | 0.83 (부트스트랩 95% 0.75–0.93) |
| 정상 대비 | 0.93 (45–70세만 0.91) |
| 파킨슨 대비 | 0.64 |
| (비교) 나이만, 정상 대비 | 0.79 |
| (비교) 속도만 / 영상 조건만, 전체 | 0.68 / 0.59 |

정상은 잘 가려내지만 파킨슨 환자의 55%를 무릎 관절염으로 오판합니다. 발표된 연구의 재현이 아니라 자체 기준 모델입니다.

## 연구와 성능이 같은가 (ST-GCN)

**코드는 같고, 성능이 같은지는 아직 확인하지 못했습니다.** 연구의 평가 데이터가 비공개라 논문 수치를 재현할 수 없고, 우리 데이터에는 UPDRS 정답이 없습니다.

| 확인한 것 | 결과 |
| --- | --- |
| 같은 입력을 넣으면 원본과 같은 출력인가 | 같음 (차이 3e-7, 연구 샘플 3개) |
| 우리 입력이 연구의 학습 입력과 비슷한가 | 비슷함 (중앙값 0.32 SD 차이, 연구 자체 샘플은 0.21 SD). 다리는 약 9% 길게 나옴 |
| 정면 반복 측정 (같은 세션) | 평균 0.17점 차이 (온몸이 보이는 구간 기준. 전 구간 기준 ICC는 0.88) |
| 측면에서 보행 방향만 바뀐 경우 | 평균 0.31점 차이 (보정 전 0.41점, ICC는 전 구간 기준 0.79). 왼쪽→오른쪽이 약 0.25점 높은 치우침은 남음 |
| 연구 카메라 각도를 5도 다르게 잡은 경우 | 영상의 26-29%에서 등급이 바뀜 |

공개된 연구 샘플 3개는 모두 카메라 쪽으로 걷는 영상입니다. 측면 결과는 보정 후에도 걷는 방향에 따라 달라지므로 참고로만 씁니다.

## 시점 변환이 얼마나 믿을 만한가

양방향으로 찍힌 87명에서 같은 사람의 평균 보행 주기가 얼마나 일치하는지 비교했습니다(체격으로 정규화, cm, 낮을수록 좋음. 같은 영상 안의 두 걸음 차이는 2.4).

| 방법 | 차이 |
| --- | --- |
| 카메라 좌표 그대로 | 44.2 |
| 좌우 반전만 | 37.7 |
| 반전 + 좌우 관절 이름 교환 | 8.5 |
| 3D 회전 (카메라 수평 가정) | 7.0 |
| 3D 회전 + 바닥 보정 (채택) | **6.3** |
| 3D 회전 + 바닥 보정 + 2D 뼈 길이 보정 | 7.3 (채택 안 함) |

- 반전은 측면을 정면으로 바꾸지 못하고, 아픈 쪽이 좌우로 뒤바뀝니다.
- 회전 계산 자체는 정확합니다. 합성 걸음에서 오차가 0이고, 남는 차이는 단안 3D 추정 오차와 걸음마다의 실제 차이입니다.
- 3D 다리가 2D 영상보다 약 9% 길게 나옵니다(투영 오차 중앙값 15px).
- 정면 영상의 이동 거리는 약 1.5배 짧게 추정됩니다. 디딘 발이 미끄러지지 않는 조건으로 측정했고(측면 0.93, 정면 1.38), 기울기 추정에만 반영합니다.
- 정면 카메라 기울기는 세션 안에서 편차 0.5도로 추정됩니다(평균 7.1도). 측면에서는 카메라 위아래 기울기를 걸음 하나만으로 알 수 없어 `lateral_deg`로 보정값을 줘야 합니다.


`koa_gait_baseline`은 발표된 연구가 아니라서 "논문이 탐지하는 것을 그대로 보여준다"는 기준에 맞지 않습니다.

관련 스크립트: `check_shift.py`(입력 분포), `check_fidelity.py`(반복성, 민감도), `compare_views.py`(반전과 3D 회전), `check_world.py`(3D 층의 효과), `explore_camera.py`·`explore_depth_scale.py`(카메라 기울기, 깊이 축척). 골반을 수평으로 맞추는 수정은 효과가 없어 적용하지 않았습니다.

## NVIDIA 추출기가 연구와 같은 결과를 내는가 (TOAGA, 판정 준비됨)

[TOAGA](https://pmc.ncbi.nlm.nih.gov/articles/PMC9272879/)는 고령자 14명의 정면·후면 영상 28개와 AlphaPose·Detectron·OpenPose 관절, Xsens 센서가 있는 공개 데이터(CC0)입니다. `python scripts/fetch_toaga.py data/toaga`로 받습니다. 참가자 통계(나이 86.7±6.2세, 키 165.6±10.0cm, 남성 21.4%)가 Sabo 2022 논문의 건강 노인군과 정확히 같아, 그 연구가 실제로 쓴 사람들과 영상입니다. 다만 임상 점수는 없습니다.

**잣대: 연구 모델에 다른 표준 추출기를 물리면 얼마나 흔들리나** (정면, 같은 프레임끼리, 사람 단위)

| 연구 모델에 넣은 관절 | 점수 차이 | 등급 변화 | 순위 상관 |
| --- | --- | --- | --- |
| AlphaPose 모델: AlphaPose → Detectron | 0.25 | 24% | 0.76 |
| AlphaPose 모델: AlphaPose → OpenPose | 0.31 | 17% | 0.21 |
| Detectron 모델: Detectron → AlphaPose | 0.16 | 18% | 0.84 |
| Detectron 모델: Detectron → OpenPose | 0.33 | 30% | 0.54 |

참고로 같은 사람의 다른 걸음은 0.16~0.17점, 폴드 10개 사이 표준편차는 0.16~0.25점 차이가 납니다. 높이가 다른 두 카메라 사이는 0.22~0.23점이지만 두 카메라가 동기화되지 않아 구간 차이도 섞여 있습니다.

**합격 기준** (`scripts/toaga_criteria.py`, NVIDIA 결과를 보기 전에 고정): 점수 차이·등급 변화·순위 상관이 AlphaPose↔Detectron 교체의 95% 범위 안이고, 걸음의 85% 이상을 영상에서 찾아야 합니다. 처음에는 범위 없이 값 하나와 비교했는데, 잣대를 정의한 추출기조차 절반은 탈락해서 NVIDIA 결과를 보기 전에 한 번 고쳤습니다.

판정기 자체 시험: 실제 TOAGA 관절로 AlphaPose(차이 0)와 Detectron은 통과, OpenPose는 탈락해야 하고 세 경우 모두 맞았습니다(`scripts/toaga_selftest.py`). 2D 직접 비교와 시점 변환 비교(아래 카메라의 3D를 가상 위 카메라로 바꿔 실제 위 카메라와 비교)는 합성 걸음으로 시험했습니다.

**다음 단계:** 팀원이 TOAGA 영상 28개를 NVIDIA로 추출하면([requests/toaga_nim_request.md](requests/toaga_nim_request.md)) 명령 한 번으로 판정됩니다.

```bash
python scripts/toaga_compare.py data/toaga/nova77 data/toaga/x/PoseTracking third_party/stgcn_parkinsonism_prediction results
```

AlphaPose·Detectron 가중치만 저장소에 있고 OpenPose 가중치는 없습니다. 정규화는 연구의 가로(1920×1080) 규칙을 씁니다. 세로 규칙은 몸을 찌그러뜨려 모델이 모든 걸음에 거의 같은 점수(1.50 ± 0.03)를 냅니다(`scripts/toaga_norm_check.py`).

## 한계

- 온몸이 화면 밖으로 나가는 프레임은 분석에서 뺍니다(파일 크기를 알 때만. `run.py --size 1280x720` 또는 데이터셋 크기 정보). 크기를 모르면 검사하지 않습니다. KOA 189개 중 135개는 앞뒤가 조금 잘려 길이가 바뀌었고, 고관절은 571개 중 11개가 60프레임 미만으로 줄었습니다.

- UPDRS 점수 정답이 없어 ST-GCN의 점수 정확도는 확인할 수 없습니다. 원본 코드와 같다는 것과 순서(NM < KOA < PD)만 확인했습니다.
- 정상군이 훨씬 젊고(평균 43.8세, PD 68.5세) 빨리 걷습니다. 나이와 속도가 질병과 섞여 있습니다.
- PD가 16명뿐이라 한 명이 바뀌면 수치가 크게 움직입니다.
- 3D 좌표는 단안 추정값입니다. 정면 영상의 이동 거리와 보폭은 미터 단위로 믿을 수 없습니다.
- 방향 판별과 무릎 플러그인은 아직 카메라가 수평이라고 가정합니다.
- 고관절 정면 영상 중 채점된 529개에서 7개는 이동 거리가 짧아 방향 판별이 의심됩니다.
- Nova-77에는 발뒤꿈치, 코, 귀가 없습니다.
- ST-GCN 가중치는 학술 비상업 용도로만 쓸 수 있어 저장소에 올리지 않습니다. `scripts/setup.py`가 원 저장소에서 받습니다.

## 출처와 라이선스

이 저장소의 코드는 수업 프로젝트용이며 따로 라이선스를 정하지 않았습니다. 상업적으로 쓸 수 없는 부분이 들어 있습니다.

| 대상 | 출처 | 라이선스 |
| --- | --- | --- |
| ST-GCN 모델 코드 이식 (`shared/stgcn.py`, `shared/sabo2d.py`, `plugins/pd_stgcn_sabo2022*`)과 가중치 | Sabo et al., IEEE JBHI 2022, [TaatiTeam/stgcn_parkinsonism_prediction](https://github.com/TaatiTeam/stgcn_parkinsonism_prediction) | UHN 학술·비상업 연구 전용 (모델 구조는 OpenMMLab mmskeleton, Apache-2.0) |
| mobile-gaitlab 모델 이식 (`plugins/gait_gaitlab2020`)과 가중치 | Kidziński et al., Nature Communications 2020, [stanfordnmbl/mobile-gaitlab](https://github.com/stanfordnmbl/mobile-gaitlab) | 코드 Apache-2.0, 학습 데이터 CC BY-NC 2.0, 스탠퍼드 특허 출원 중 |
| 걸음 시점 규칙 (`plugins/gait_events_zeni2008`) | Zeni et al., Gait & Posture 2008 | 논문의 계산식 |
| `samples/`, `demo/`, `results/`의 KOA-PD-NM 유래 값 | [Mendeley Data 44pfnysy89](https://data.mendeley.com/datasets/44pfnysy89/1) | CC BY 4.0 |
| 같은 곳의 Hip_OA 유래 값 | [figshare 14420645](https://doi.org/10.6084/m9.figshare.14420645) | CC BY 4.0 |
| TOAGA 유래 값 | [figshare (Toronto Older Adults Gait Archive)](https://doi.org/10.6084/m9.figshare.c.5515953.v1) | CC0 |

관절 좌표는 위 데이터셋의 영상에서 NVIDIA NIM 3D Body Pose(Nova-77, NIM 1.0.1)로 추출했습니다. 원 데이터 제공자와 연구 저자는 이 저장소를 보증하지 않습니다.
