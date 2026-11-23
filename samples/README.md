# 예시 파일

NVIDIA NIM 3D Body Pose(Nova-77, NIM 1.0.1)로 추출한 실제 보행 기록 3개입니다. 용량 때문에 gzip으로 압축했고, `core/nova77.load`가 그대로 읽습니다. 시연 화면(`demo/index.html`)의 예시 중 3개와 같은 파일입니다.

| 파일 | 데이터셋 | 집단 | 촬영 방향 | 영상 크기 (`--size`) |
| --- | --- | --- | --- | --- |
| `004_NM_01.jsonl.gz` | KOA-PD-NM | 정상 | 측면 | 1280x720 |
| `001_PD_01_MD.jsonl.gz` | KOA-PD-NM | 파킨슨병, 중등도 | 측면 | 1280x720 |
| `HOA001_M0_gait03.60948372.jsonl.gz` | Hip_OA | 고관절 관절염, 수술 전 | 정면 (카메라 반대쪽으로) | 644x484 |

```bash
python scripts/run.py samples/004_NM_01.jsonl.gz samples/001_PD_01_MD.jsonl.gz --size 1280x720
python scripts/run.py samples/HOA001_M0_gait03.60948372.jsonl.gz --size 644x484
```

## 출처와 라이선스

원본 영상은 아래 공개 데이터셋에서 왔고, 여기 있는 것은 영상에서 추출한 관절 좌표입니다(변경 사항: 영상을 30fps, 높이 최대 720px로 변환한 뒤 관절 좌표 추출).

- KOA-PD-NM: *Gait Dataset for Knee Osteoarthritis and Parkinson's Disease Analysis With Severity Levels*, Mendeley Data, https://data.mendeley.com/datasets/44pfnysy89/1, CC BY 4.0
- Hip_OA: *Dataset of gait analysis data of hip osteoarthritis patients before and 6 month after arthroplasty and asymptomatic volunteers*, figshare, https://doi.org/10.6084/m9.figshare.14420645, CC BY 4.0

원 데이터 제공자는 이 저장소를 보증하지 않습니다.
