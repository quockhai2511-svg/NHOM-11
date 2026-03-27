"""
Nhận diện bàn tay bằng MediaPipe + OpenCV
Cài đặt: pip install mediapipe opencv-python
"""

import cv2
import mediapipe as mp
import time

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,      # False = video stream (nhanh hơn)
    max_num_hands=2,              # Số bàn tay tối đa nhận diện
    min_detection_confidence=0.7, # Ngưỡng tin cậy phát hiện (0.0 - 1.0)
    min_tracking_confidence=0.5   # Ngưỡng tin cậy theo dõi
)

# Tên 21 điểm mốc (landmark) của bàn tay
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Đầu ngón tay (tip landmark index)
FINGER_TIPS = [4, 8, 12, 16, 20]  # Ngón cái, trỏ, giữa, áp út, út
FINGER_PIPS = [3, 6, 10, 14, 18]  # Khớp giữa tương ứng


def count_fingers(hand_landmarks, hand_label):
    """Đếm số ngón tay đang giơ lên"""
    count = 0
    lm = hand_landmarks.landmark

    # Ngón cái: so sánh theo trục X (trái/phải tùy tay)
    if hand_label == "Right":
        if lm[FINGER_TIPS[0]].x < lm[FINGER_PIPS[0]].x:
            count += 1
    else:
        if lm[FINGER_TIPS[0]].x > lm[FINGER_PIPS[0]].x:
            count += 1

    # 4 ngón còn lại: so sánh theo trục Y
    for i in range(1, 5):
        if lm[FINGER_TIPS[i]].y < lm[FINGER_PIPS[i]].y:
            count += 1

    return count


def get_gesture(finger_count, hand_label):
    """Nhận biết cử chỉ cơ bản"""
    gestures = {
        0: "✊ Nắm tay",
        1: "☝️ Một ngón",
        2: "✌️ Hai ngón (Hòa bình)",
        3: "🤟 Ba ngón",
        4: "🖐️ Bốn ngón",
        5: "✋ Xòe tay"
    }
    return gestures.get(finger_count, f"{finger_count} ngón")


def draw_info(frame, fps, hand_count):
    """Vẽ thông tin lên màn hình"""
    h, w = frame.shape[:2]

    # Nền mờ cho thông tin
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Ban tay: {hand_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Hướng dẫn góc trên phải
    cv2.putText(frame, "Nhan Q de thoat", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def main():
    # Mở camera (0 = camera mặc định)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Không thể mở camera! Kiểm tra lại thiết bị.")
        return

    # Cài đặt độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ Đang chạy nhận diện bàn tay...")
    print("   Nhấn Q để thoát")
    print("   Nhấn S để chụp ảnh")
    print("-" * 40)

    prev_time = time.time()
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame từ camera!")
            break

        # Lật ngược ảnh để giống gương
        frame = cv2.flip(frame, 1)

        # Chuyển BGR → RGB (MediaPipe yêu cầu RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Xử lý nhận diện
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hand_count = 0

        # Vẽ kết quả nếu phát hiện bàn tay
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)

            for idx, (hand_lm, hand_info) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                hand_label = hand_info.classification[0].label  # "Left" hoặc "Right"
                confidence = hand_info.classification[0].score

                # Vẽ các điểm mốc và đường kết nối
                mp_draw.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )

                # Tính vị trí bounding box
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_lm.landmark]
                y_coords = [lm.y * h for lm in hand_lm.landmark]
                x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
                y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20

                # Đảm bảo không vượt ra ngoài khung hình
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # Màu khác nhau cho tay trái/phải
                color = (0, 255, 100) if hand_label == "Right" else (255, 100, 0)

                # Vẽ bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Đếm ngón tay và nhận cử chỉ
                finger_count = count_fingers(hand_lm, hand_label)
                gesture = get_gesture(finger_count, hand_label)

                # Nhãn phía trên bounding box
                label_text = f"{'Trai' if hand_label == 'Left' else 'Phai'} ({confidence:.0%})"
                label_bg_y = y_min - 10 if y_min > 40 else y_max + 30
                cv2.rectangle(frame, (x_min, label_bg_y - 22), (x_min + 200, label_bg_y + 4), color, -1)
                cv2.putText(frame, label_text, (x_min + 4, label_bg_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

                # Cử chỉ phía dưới bounding box
                gesture_y = y_max + 30
                cv2.putText(frame, gesture, (x_min, gesture_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                # In thông tin ra terminal
                print(f"\r🖐 Tay {idx+1}: {hand_label} | {gesture} | Tin cậy: {confidence:.0%}   ", end="")

        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # Vẽ thông tin tổng quan
        draw_info(frame, fps, hand_count)

        # Hiển thị khung hình
        cv2.imshow("Nhan Dien Ban Tay - MediaPipe", frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q hoặc ESC để thoát
            break
        elif key == ord('s'):  # S để chụp ảnh
            screenshot_count += 1
            filename = f"hand_screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n📸 Đã lưu: {filename}")

    print("\n\n✅ Đã thoát chương trình.")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()