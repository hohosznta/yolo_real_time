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
from kakao_sender import KakaoSender
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
    kakao_sender = KakaoSender()

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
                for warning in warnings:
                    print(f"  - {warning.get('type', 'unknown')}: {warning.get('message', '')}")

                # 알림 간격 체크 (5분)
                if Utils.should_notify(int(ts), last_notification_time):
                    print("📱 카카오톡 알림 전송 중...")
                    success = await kakao_sender.send_violation_alert(
                        site=site,
                        stream_name=stream_name,
                        warnings=warnings,
                        detection_time=detection_time,
                    )

                    if success:
                        print(" 카카오톡 알림 전송 완료")
                        last_notification_time = int(ts)
                    else:
                        print(" 카카오톡 알림 전송 실패")
                else:
                    remaining = 300 - (int(ts) - last_notification_time)
                    print(f" 다음 알림까지: {remaining}초")

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
        await kakao_sender.close()
        print("✅ 종료 완료")


if __name__ == "__main__":
    print("BlueGuard 간단 실행 모드")

    # 카카오 토큰 확인
    if not os.getenv('KAKAO_ACCESS_TOKEN'):
        print("카카오 토큰이 설정되지 않았습니다.")
        print(".env 파일에 KAKAO_ACCESS_TOKEN을 설정하세요.")
        print()

    # 실행
    asyncio.run(run_simple_stream())