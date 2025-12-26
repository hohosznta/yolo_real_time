#!/usr/bin/env python3
"""
간단한 실행 스크립트 - 웹캠으로 바로 테스트
"""

import asyncio
import os
import time
from datetime import datetime

from dotenv import load_dotenv

from danger_detector import DangerDetector
from kakao_sender import KakaoNotifier
from live_stream_detection import LiveStreamDetector
from utils import Utils

# 환경변수 로드
load_dotenv()


async def run_simple_stream():
    """단순 스트림 실행"""

    # 설정
    site = "테스트현장"
    stream_name = "웹캠"

    print(f"🎥 스트리밍 시작: {site}/{stream_name}")
    print("종료하려면 Ctrl+C를 누르세요\n")

    # 컴포넌트 초기화
    live_stream_detector = LiveStreamDetector(
        source = 0,
        model_path = 'models/pt/best_yolo11n.pt'
    )
    danger_detector = DangerDetector({
    'detect_no_safety_vest_or_helmet': True,
    'detect_near_machinery_or_vehicle': True,
    'detect_in_restricted_area': True,
    'detect_in_utility_pole_restricted_area': False,
    'detect_machinery_close_to_pole': False,
})
    kakao_notifier = KakaoNotifier()

    last_notification_time = 0
    frame_count = 0
    total_detections = 0
    total_warnings = 0
    start_time = time.time()

    try:
        # 프레임 처리 루프
        for ids, datas, frame, ts in live_stream_detector.generate_detections():
            frame_count += 1

            # 5프레임마다 처리 (성능 최적화)
            if frame_count % 5 != 0:
                continue

            start = time.time()
            detection_time = datetime.fromtimestamp(int(ts))

            # YOLO 감지
            print(f"🔍 프레임 {frame_count}: 감지 중...", end='')

            # Class mapping for display
            class_names = {
                0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask',
                4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone',
                7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'
            }

            # 감지된 객체 출력
            if datas:
                detected_objects = {}
                for data in datas:
                    if len(data) > 5:
                        class_id = int(data[5])
                        label = class_names.get(class_id, f'unknown_{class_id}')
                        if label not in detected_objects:
                            detected_objects[label] = 0
                        detected_objects[label] += 1

                total_detections += len(datas)
                objects_str = ", ".join([f"{count} {obj}" for obj, count in detected_objects.items()])
                print(f" → 감지: {objects_str}", end='')
            else:
                print(f" → 감지된 객체 없음", end='')

            # 상태 정보 추가
            elapsed = int(time.time() - start_time)
            print(f" | 총 감지: {total_detections} | 경고: {total_warnings} | 경과: {elapsed}초")
            print(datas)

            # 위험 감지 - datas를 그대로 전달
            warnings, cone_polys, pole_polys = danger_detector.detect_danger(datas)

            # 위험 감지 시 알림
            if warnings:
                total_warnings += len(warnings)
                print(f"⚠️ 위험 감지! {len(warnings)}건의 위반 사항")
                for warning_type, warning_data in warnings.items():
                    # Convert warning type to readable message
                    message = warning_type.replace('_', ' ').replace('warning ', '')
                    count = warning_data.get('count', 0)
                    print(f"  - {message}: {count}명/개")

                # 알림 간격 체크 (30초)
                if Utils.should_notify(int(ts), last_notification_time):
                    print("📱 카카오톡 알림 전송 중...")

                    # warnings를 List[Dict] 형태로 변환
                    warning_list = []
                    warning_descriptions = {
                        'warning_no_hardhat': '안전모 미착용,  산업안전보건법 제175조 제6항에 의거 300만원 이하의 과태료에 처해질 수 있습니다.',
                        'warning_no_safety_vest': '안전조끼 미착용, 산업안전보건법 제175조 제6항에 의거 300만원 이하의 과태료에 처해질 수 있습니다.',
                        'warning_close_to_machinery': '중장비 근접 위험',
                        'warning_close_to_vehicle': '차량 근접 위험',
                        'warning_people_in_controlled_area': '통제구역 침입',
                        'warning_people_in_utility_pole_controlled_area': '전신주 통제구역 침입',
                        'detect_machinery_close_to_pole': '전신주 근처 중장비 위험'
                    }

                    for warning_type, warning_data in warnings.items():
                        count = warning_data.get('count', 0)
                        description = warning_descriptions.get(warning_type, warning_type.replace('_', ' '))
                        warning_list.append({
                            'type': warning_type,
                            'description': f"{description}"
                        })

                    result = await kakao_notifier.send_violation_alert(
                        site=site,
                        stream_name=stream_name,
                        warnings=warning_list,
                        detection_time=detection_time,     
                    )

                    if isinstance(result, tuple):
                        success, error_msg = result
                    else:
                        success = result
                        error_msg = "상세 메시지 없음"

                    if success:
                        print(f"✅ 카카오톡 알림 전송 완료: {error_msg}")
                        last_notification_time = int(ts)
                    else:
                        print(f"❌ 카카오톡 알림 전송 실패: {error_msg}")
                else:
                    remaining = 300 - (int(ts) - last_notification_time)
                    print(f"⏳ 다음 알림까지: {remaining}초")

            # 처리 시간 표시
            proc_time = time.time() - start
            print(f"    처리 시간: {proc_time:.3f}초")

    except KeyboardInterrupt:
        print("\n\n🛑 사용자 중단")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        # 정리
        print("🧹 리소스 정리 중...")
        live_stream_detector.release_resources()
        await kakao_notifier.close()
        print("✅ 종료 완료")


if __name__ == "__main__":
    print("BlueGuard 간단 실행 모드")

    # 카카오 설정 확인
    if not os.getenv('KAKAO_REST_API_KEY'):
        print("⚠️ 카카오 REST API 키가 설정되지 않았습니다.")
        print(".env 파일에 KAKAO_REST_API_KEY를 설정하세요.")
        print()

    if not os.path.exists("kakao_access_token_data.json") and not os.getenv('KAKAO_AUTH_CODE'):
        print("⚠️ 카카오 인증이 필요합니다.")
        print("1. 카카오 개발자 사이트에서 인증 코드를 받으세요")
        print("2. .env 파일에 KAKAO_AUTH_CODE를 설정하세요")
        print("3. 또는 기존 kakao_access_token_data.json 파일을 복사하세요")
        print()

    # 실행
    asyncio.run(run_simple_stream())