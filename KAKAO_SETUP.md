# 카카오톡 알림 설정 가이드

## 1. 카카오 개발자 설정

### 1.1 카카오 개발자 계정 생성
1. [카카오 개발자 사이트](https://developers.kakao.com) 접속
2. 카카오 계정으로 로그인
3. 애플리케이션 생성

### 1.2 카카오톡 메시지 API 설정
1. 애플리케이션 설정 > 카카오 로그인 활성화
2. Redirect URI 설정: `http://localhost:3000/oauth`
3. 동의 항목에서 "카카오톡 메시지 전송" 설정

### 1.3 Access Token 발급
```bash
# 인가 코드 받기 (브라우저에서 실행)
https://kauth.kakao.com/oauth/authorize?client_id={APP_KEY}&redirect_uri={REDIRECT_URI}&response_type=code&scope=talk_message

# Access Token 받기
curl -X POST https://kauth.kakao.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "client_id={APP_KEY}" \
  -d "redirect_uri={REDIRECT_URI}" \
  -d "code={AUTHORIZATION_CODE}"
```

## 2. BlueGuard 설정

### 2.1 환경 변수 설정
`.env` 파일을 수정:
```env
KAKAO_ACCESS_TOKEN=your_access_token_here
KAKAO_API_URL=https://kapi.kakao.com/v2/api/talk/memo/default/send
```

### 2.2 기능 활성화
`main.py`에서 이미 설정됨:
```python
violation_sender = ViolationSender(
    api_url=os.getenv('VIOLATION_RECORD_API_URL') or '',
    enable_kakao=True,  # 카카오톡 알림 활성화
)
```

## 3. 테스트

### 3.1 단독 테스트
```bash
python test_kakao.py
```

### 3.2 전체 시스템 테스트
```bash
python main.py --config test_config.json
```

## 4. 알림 메시지 형식

위반 감지 시 다음과 같은 메시지가 전송됩니다:

```
⚠️ 안전 위반 감지!

📍 현장: 건설현장 A
📹 카메라: 카메라 #1
🕐 시간: 2024-12-24 10:30:45

위반 내역:
1. danger_zone_violation
2. no_helmet
3. proximity_violation

즉시 확인이 필요합니다.
```

## 5. 주의사항

1. **Access Token 유효기간**: 약 12시간 (갱신 필요)
2. **메시지 전송 제한**: 나에게 보내기는 제한 없음, 친구에게는 별도 권한 필요
3. **알림 간격**: 스팸 방지를 위해 5분 간격으로 제한 가능

## 6. 문제 해결

### Token 만료 오류
```
❌ Kakao notification failed: {"msg":"this access token is already expired","code":-401}
```
→ 새로운 Access Token 발급 필요

### 권한 오류
```
❌ Kakao notification failed: {"msg":"insufficient scopes","code":-402}
```
→ 카카오톡 메시지 전송 권한 확인 필요