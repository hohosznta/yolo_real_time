import requests
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class KakaoNotifier:
    """카카오톡 알림 전송 클래스 - OAuth 인증 방식"""

    def __init__(self, rest_api_key: str = None, auth_code: str = None):
        """
        초기화

        Args:
            rest_api_key: 카카오 REST API 키
            auth_code: 인증 코드 (최초 인증시 필요)
        """
        # 환경변수에서 로드
        load_dotenv()
        self.rest_api_key = rest_api_key or os.getenv("KAKAO_REST_API_KEY")
        self.auth_code = auth_code or os.getenv("KAKAO_AUTH_CODE")
        self.redirect_uri = 'https://example.com/oauth'
        self.token_file = "kakao_access_token_data.json"
        self.send_url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
        self.token_info = {}

        # 알림 간격 제한 (스팸 방지)
        self.last_notification_time = 0
        self.notification_interval = 600  # 10분

        logging.info(f"KakaoNotifier 초기화: API키={'있음' if self.rest_api_key else '없음'}, 인증코드={'있음' if self.auth_code else '없음'}")

        if os.path.exists(self.token_file):
            logging.info("기존 토큰 파일 발견, 로드 중...")
            self.load_token()
            if self.is_token_expired():
                logging.info("토큰이 만료됨. 갱신 필요.")
                self.get_access_token()
        elif self.auth_code:
            logging.info("토큰 파일이 없음. 새로 발급 중...")
            self.get_access_token()
        else:
            self.get_access_token()
            logging.warning("토큰 파일과 인증 코드가 모두 없습니다.")

    def get_access_token(self):
        """최초 액세스 토큰 발급"""
        URL = 'https://kauth.kakao.com/oauth/token'

        data = {
            'grant_type': 'authorization_code',
            'client_id': self.rest_api_key,
            'redirect_uri': self.redirect_uri,
            'code': self.auth_code
        }

        try:
            logging.info("카카오 서버에 토큰 요청 중...")
            res = requests.post(URL, data=data)
            logging.info(f"응답 상태: {res.status_code}")

            res_json = res.json()

            if "access_token" not in res_json:
                logging.error(f"토큰 발급 실패: {res_json}")
                return False

            self.token_info = res_json
            # 만료 시간 추가
            expires_in = res_json.get("expires_in", 21600)
            self.token_info["expires_at"] = (datetime.now() + timedelta(seconds=expires_in)).isoformat()

            self.save_token()
            logging.info("토큰 발급 및 저장 완료!")
            return True

        except Exception as e:
            logging.error(f"토큰 발급 중 오류: {e}")
            return False

    def is_token_expired(self) -> bool:
        """토큰 만료 확인"""
        if not self.token_info.get("expires_at"):
            return True

        expires_at = datetime.fromisoformat(self.token_info["expires_at"])
        # 10분 여유를 두고 갱신
        return datetime.now() >= (expires_at - timedelta(minutes=10))

    def save_token(self):
        """토큰 정보를 파일에 저장"""
        try:
            with open(self.token_file, "w", encoding='utf-8') as f:
                json.dump(self.token_info, f, indent=2, ensure_ascii=False)
            logging.debug("토큰 파일 저장 완료")
        except Exception as e:
            logging.error(f"토큰 저장 실패: {e}")

    def load_token(self):
        """파일에서 토큰 정보 로드"""
        try:
            with open(self.token_file, "r", encoding='utf-8') as f:
                self.token_info = json.load(f)
            logging.debug("토큰 로드 완료")
        except Exception as e:
            logging.error(f"토큰 로드 실패: {e}")
            self.token_info = {}

    def send_message(self, text: str) -> tuple[bool, str]:
        """
        카카오톡 메시지 전송

        Args:
            text: 전송할 메시지

        Returns:
            (성공 여부, 메시지)
        """
        # 토큰 확인
        if not self.token_info.get("access_token"):
            error_msg = "유효한 액세스 토큰이 없습니다."
            logging.error(error_msg)
            return False, error_msg

        # 토큰 만료 확인 및 갱신
        if self.is_token_expired():
            logging.info("토큰이 만료되어 갱신을 시도합니다.")
            if not self.get_access_token():
                error_msg = "토큰 갱신 실패"
                return False, error_msg

        headers = {
            "Authorization": f"Bearer {self.token_info['access_token']}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        template = {
            "object_type": "text",
            "text": text,
            "link": {
                "web_url": "https://blueguard.site",
                "mobile_web_url": "https://blueguard.site"
            },
            "button_title": "자세히 보기"
        }

        data = {
            "template_object": json.dumps(template, ensure_ascii=False)
        }

        try:
            logging.debug("메시지 전송 중...")
            response = requests.post(self.send_url, headers=headers, data=data)

            if response.status_code != 200:
                error_msg = f"카카오톡 전송 실패: 상태코드={response.status_code}, 응답={response.text}"
                logging.error(error_msg)
                return False, error_msg
            else:
                success_msg = "카카오톡 메시지 전송 완료!"
                logging.info(success_msg)
                return True, success_msg

        except requests.RequestException as e:
            error_msg = f"네트워크 오류: {e}"
            logging.error(error_msg)
            return False, error_msg

    async def send_violation_alert(
        self,
        site: str,
        stream_name: str,
        warnings: List[Dict],
        detection_time: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """
        위반 알림 전송 (비동기 래퍼)

        Args:
            site: 현장 이름
            stream_name: 카메라/스트림 이름
            warnings: 위반 사항 리스트
            detection_time: 감지 시간

        Returns:
            (전송 성공 여부, 메시지)
        """
        # 입력 데이터 유효성 검사
        if not isinstance(warnings, list):
            error_msg = f"warnings는 리스트여야 합니다. 현재 타입: {type(warnings)}"
            logging.error(error_msg)
            return False, error_msg

        # 메시지 생성
        message = self._create_message(site, stream_name, warnings, detection_time)

        # 동기 함수 호출
        return self.send_message(message)

    def _create_message(
        self,
        site: str,
        stream_name: str,
        warnings: List[Dict],
        detection_time: Optional[datetime]
    ) -> str:
        """알림 메시지 생성"""
        message = "⚠️ BlueGuard 위험 감지 알림\n"
        message += "━━━━━━━━━━━━━━\n\n"

        message += f"📍 현장: {site}\n"
        message += f"📹 카메라: {stream_name}\n"

        if detection_time:
            message += f"🕐 시간: {detection_time.strftime('%H:%M:%S')}\n"

        message += "\n🚨 위반 사항:\n"

        # 위반 사항 표시 (최대 5개)
        for i, warning in enumerate(warnings[:5], 1):
            if isinstance(warning, dict):
                warning_type = warning.get('type', 'unknown')
                description = warning.get('description', '')

                # 위반 타입별 이모지
                emoji = {
                    'warning_no_hardhat': '⛑️',
                    'warning_no_safety_vest': '🦺',
                    'warning_close_to_machinery': '⚠️',
                    'warning_close_to_vehicle': '🚗',
                    'warning_people_in_controlled_area': '⛔',
                    'warning_people_in_utility_pole_controlled_area': '⚡',
                    'detect_machinery_close_to_pole': '🏗️'
                }.get(warning_type, '❗')

                message += f"{i}. {emoji} {description}\n"

        if len(warnings) > 5:
            message += f"... 외 {len(warnings) - 5}건\n"

        message += "\n즉시 현장을 확인해주세요!"
        return message

    async def close(self):
        """리소스 정리 (호환성을 위해 유지)"""
        pass


if __name__ == "__main__":
    # 테스트 코드
    import asyncio

    async def test():
        # 테스트 위반 데이터
        test_warnings = [
            {"type": "warning_no_hardhat", "description": "안전모 미착용 (2명)"},
            {"type": "warning_close_to_machinery", "description": "중장비 근접 위험 (1명)"},
        ]

        # 알림 전송
        notifier = KakaoNotifier()
        success, message = await notifier.send_violation_alert(
            "테스트 현장",
            "카메라 #1",
            test_warnings,
            datetime.now(),
            check_interval=False  # 테스트시 간격 체크 비활성화
        )

        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")

    # 실행
    asyncio.run(test())