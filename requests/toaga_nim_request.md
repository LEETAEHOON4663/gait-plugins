# TOAGA 영상 NVIDIA 추출 요청

연구 모델(Sabo 2022 ST-GCN)이 NVIDIA 스켈레톤을 넣었을 때도 원래 연구와 같은 점수를 내는지 확인하려고 합니다. 걸음 영상 28개를 지난번과 **같은 방식으로** 추출해 주세요.

## 받을 영상

- 데이터: Toronto Older Adults Gait Archive (Figshare, CC0). https://doi.org/10.6084/m9.figshare.c.5515953.v1
- `Videos.zip` (2.35GB) 안의 `OAW01-top.mp4` ~ `OAW14-bottom.mp4`, 28개
- 작업 서버에도 있습니다: `<작업 폴더>/data/toaga/x/Videos/Videos/`

## 추출 조건

- 지난번과 같은 NIM 1.0.1, 같은 `extract_dataset_nova77_v0_1.py` 설정
- 영상 **전체**를 자르지 말고 추출 (걷는 구간은 제가 자동으로 찾습니다)
- 세로 영상(1080×1920)입니다. 가로세로 비율만 유지하면 크기는 상관없습니다
- 한 프레임에 가장 큰 사람 한 명만 (지난번과 동일). 관찰자가 화면에 잡히면 알려 주세요
- `bbox` 없이 지난번과 같은 형식이면 됩니다

## 보내 줄 것

- `OAW01-top.jsonl` 처럼 **영상 이름 그대로** 28개 JSONL
- `batch_status`처럼 실패한 영상 목록
- 올릴 곳: 작업 서버 `<작업 폴더>/data/toaga/nova77/` (또는 Drive 폴더)

## 받으면 하는 일

```bash
cd <작업 폴더>
.venv/bin/python -I gait_plugins/scripts/toaga_compare.py data/toaga/nova77 data/toaga/x/PoseTracking third_party/stgcn_parkinsonism_prediction results
```

합격 기준은 NVIDIA 결과를 보기 전에 `scripts/toaga_criteria.py`에 고정해 두었습니다.
