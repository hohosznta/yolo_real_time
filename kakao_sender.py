"""
카카오톡 알림 전송 모듈
위반 사항 감지 시 카카오톡으로 알림을 보내는 간단한 모듈
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional

import httpx
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


class KakaoSender:
    """카카오톡 알림 전송 클래스"""

    def __init__(
        self,
        access_token: str | None = None,
        notification_interval: int = 300,  # 5분 간격
    ) -> None:
        """
        초기화

        Args:
            access_token: 카카오 API 액세스 토큰
            notification_interval: 알림 최소 간격 (초)
        """
        self.access_token = access_token or os.getenv('KAKAO_ACCESS_TOKEN')
        self.api_url = os.getenv(
            'KAKAO_API_URL',
            'https://kapi.kakao.com/v2/api/talk/memo/default/send'
        )
        self.notification_interval = notification_interval
        self.last_notification_time = 0

        # HTTP 클라이언트
        self._client: httpx.AsyncClient | None = None

        # 로깅 설정
        logging.getLogger('httpx').setLevel(logging.WARNING)

    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 가져오기"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10),
                limits=httpx.Limits(max_keepalive_connections=5)
            )
        return self._client

    async def send_violation_alert(
        self,
        site: str,
        stream_name: str,
        warnings: List[Dict],
        detection_time: datetime | None = None,
        check_interval: bool = True,
    ) -> bool:
        """
        위반 알림 전송

        Args:
            site: 현장 이름
            stream_name: 카메라/스트림 이름
            warnings: 위반 사항 리스트
            detection_time: 감지 시간
            check_interval: 알림 간격 체크 여부

        Returns:
            전송 성공 여부
        """
        # 액세스 토큰 확인
        if not self.access_token:
            logging.error("카카오 액세스 토큰이 없습니다. KAKAO_ACCESS_TOKEN 환경변수를 설정하세요.")
            return False

        # 알림 간격 체크 (스팸 방지)
        if check_interval:
            import time
            current_time = time.time()
            if current_time - self.last_notification_time < self.notification_interval:
                remaining = self.notification_interval - (current_time - self.last_notification_time)
                logging.info(f"알림 대기 중... (다음 알림까지 {remaining:.0f}초)")
                return False
            self.last_notification_time = current_time

        # 메시지 생성
        message = self._create_message(site, stream_name, warnings, detection_time)

        # 카카오톡 템플릿
        template = {
            "object_type": "text",
            "text": message,
            "link": {
                "web_url": f"http://blueguard.site/{site}/{stream_name}",
                "mobile_web_url": f"http://blueguard.site/{site}/{stream_name}"
            },
            "button_title": "실시간 확인"
        }

        # API 요청
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
        }

        data = {
            "template_object": json.dumps(template, ensure_ascii=False)
        }

        try:
            client = await self._get_client()
            response = await client.post(
                self.api_url,
                headers=headers,
                data=data
            )
            response.raise_for_status()

            logging.info(f"✅ 카카오톡 알림 전송 성공: {site}/{stream_name}")
            return True

        except httpx.HTTPStatusError as e:
            logging.error(f"❌ 카카오톡 API 오류: {e.response.text if e.response else str(e)}")
            return False
        except Exception as e:
            logging.error(f"❌ 알림 전송 실패: {e}")
            return False

    def _create_message(
        self,
        site: str,
        stream_name: str,
        warnings: List[Dict],
        detection_time: datetime | None,
    ) -> str:
        """알림 메시지 생성"""
        message = "⚠️ 위험 감지 알림\n"
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
                    'danger_zone_violation': '⛔',
                    'no_helmet': '⛑️',
                    'proximity_violation': '🚧',
                    'no_safety_vest': '🦺',
                }.get(warning_type, '❗')

                message += f"  {emoji} {description or warning_type}\n"

        if len(warnings) > 5:
            message += f"  ... 외 {len(warnings)-5}건\n"

        message += "\n즉시 확인 바랍니다!"

        return message

    async def close(self) -> None:
        """클라이언트 정리"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.close()


# 간편 사용 함수
async def send_kakao_alert(
    site: str,
    stream_name: str,
    warnings: List[Dict],
    detection_time: datetime | None = None,
) -> bool:
    """
    카카오톡 알림 전송 (간편 함수)

    사용 예:
        await send_kakao_alert("현장A", "카메라1", warnings)
    """
    sender = KakaoSender()
    try:
        return await sender.send_violation_alert(
            site, stream_name, warnings, detection_time
        )
    finally:
        await sender.close()


if __name__ == "__main__":
    # 테스트 코드
    import asyncio

    async def test():
        # 테스트 위반 데이터
        test_warnings = [
            {"type": "danger_zone_violation", "description": "제한구역 침입"},
            {"type": "no_helmet", "description": "안전모 미착용"},
            {"type": "proximity_violation", "description": "중장비 근접 위험"},
        ]

        # 알림 전송
        success = await send_kakao_alert(
            site="테스트 현장",
            stream_name="정문 카메라",
            warnings=test_warnings,
            detection_time=datetime.now()
        )

        if success:
            print("✅ 테스트 알림 전송 성공!")
        else:
            print("❌ 테스트 알림 전송 실패")

    print("카카오톡 알림 테스트 시작...")
    asyncio.run(test())