# BlueGuard - 실시간 안전 모니터링 시스템

YOLO 기반 실시간 객체 감지 및 위험 알림 시스템

## 주요 기능

- 🎥 실시간 비디오 스트림 모니터링 (웹캠, RTSP, 비디오 파일)
- 🔍 YOLO v8 기반 객체 감지 (사람, 차량, 트럭 등)
- ⚠️ 위험 구역 침입 감지 및 경고
- 📱 카카오톡 실시간 알림
- 📊 로깅 및 모니터링

## 설치

1. 필요한 패키지 설치:
```bash
pip install opencv-python ultralytics python-dotenv aiohttp numpy
```

2. YOLO 모델 다운로드:
   - `models/` 폴더에 YOLO 모델 파일 배치

3. 환경 설정:
   - `.env` 파일에 카카오 토큰 설정
   - 자세한 설정은 `KAKAO_SETUP.md` 참조

## 사용법

### 간단 실행 (웹캠 테스트)
```bash
python simple_run.py
```

### 설정 파일 사용
```bash
python main.py --config test_config.json
```

## 프로젝트 구조

- `simple_run.py` - 간단한 웹캠 테스트
- `main.py` - 메인 실행 파일
- `stream_capture.py` - 비디오 스트림 캡처
- `live_stream_detection.py` - YOLO 객체 감지
- `danger_detector.py` - 위험 구역 감지
- `kakao_sender.py` - 카카오톡 알림
- `monitor_logger.py` - 로깅 시스템
- `utils.py` - 유틸리티 함수

## 문서

- [카카오톡 설정 가이드](KAKAO_SETUP.md)
- [간단 사용 가이드](SIMPLE_USAGE.md)

## License

MIT