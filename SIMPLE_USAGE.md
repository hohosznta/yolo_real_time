# 🚨 BlueGuard 카카오톡 알림 간단 사용법

## 📋 빠른 시작

### 1. 카카오 토큰 발급
```bash
# 1단계: 브라우저에서 인가 코드 받기
https://kauth.kakao.com/oauth/authorize?client_id=YOUR_APP_KEY&redirect_uri=http://localhost:3000&response_type=code&scope=talk_message

# 2단계: 토큰 발급 (터미널에서)
curl -X POST https://kauth.kakao.com/oauth/token \
  -d "grant_type=authorization_code" \
  -d "client_id=YOUR_APP_KEY" \
  -d "code=AUTHORIZATION_CODE"
```

### 2. 환경 설정 (.env)
```env
KAKAO_ACCESS_TOKEN=발급받은_토큰
KAKAO_API_URL=https://kapi.kakao.com/v2/api/talk/memo/default/send
```

### 3. 실행
```bash
# 카카오톡 알림 테스트
python kakao_sender.py

# 전체 시스템 실행 (YOLO 감지 + 카카오톡)
python main.py
```

## 🏗️ 간소화된 구조

```
위험 감지 (YOLO)
    ↓
위반 규칙 체크
    ↓
카카오톡 알림 전송
```

### 제거된 기능
- ❌ 백엔드 API 서버 연동
- ❌ Redis 스트리밍
- ❌ 복잡한 인증 시스템
- ❌ 데이터베이스 저장

### 유지된 핵심 기능
- ✅ YOLO 실시간 감지
- ✅ 위험 구역 판단
- ✅ 카카오톡 즉시 알림
- ✅ 스팸 방지 (5분 간격)

## 📱 알림 예시

```
⚠️ 위험 감지 알림
━━━━━━━━━━━━━━
📍 현장: 공사현장 A
📹 카메라: 정문 CCTV
🕐 시간: 14:30:25

🚨 위반 사항:
  ⛔ 제한구역 침입
  ⛑️ 안전모 미착용
  🚧 중장비 근접 위험

즉시 확인 바랍니다!
```

## 🔧 커스터마이징

### 위험 구역 설정 (danger_detector.py)
```python
self.danger_zones = [
    {'name': '크레인 작업구역', 'bbox': [100, 100, 400, 400]},
    {'name': '굴착 구역', 'bbox': [500, 100, 800, 400]},
]
```

### 알림 간격 변경 (kakao_sender.py)
```python
# 10분으로 변경
kakao_sender = KakaoSender(notification_interval=600)
```

### 감지 클래스 변경 (live_stream_detection.py)
```python
self.danger_classes = {
    0: "person",     # 사람
    2: "car",        # 차량
    7: "truck",      # 트럭
    39: "helmet",    # 안전모 (커스텀 모델)
}
```

## ⚡ 성능 팁

1. **프레임 스킵**: 5프레임마다 처리
2. **YOLO 모델**: YOLOv8n (nano) 사용으로 빠른 처리
3. **알림 제한**: 스팸 방지로 5분 간격

## 🐛 문제 해결

### "카카오 액세스 토큰이 없습니다"
→ `.env` 파일에 `KAKAO_ACCESS_TOKEN` 설정

### "this access token is already expired"
→ 토큰 재발급 필요 (12시간마다)

### 알림이 오지 않음
→ 카카오톡 > 더보기 > 설정 > 카카오톡 실험실 > 나에게 보내기 활성화

## 📞 지원

문제가 있으면 이슈를 등록해주세요!