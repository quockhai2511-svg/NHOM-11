"""
Nhận diện bàn tay + Gửi Gmail khi đưa cử chỉ V (2 ngón)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cài đặt:
    pip install mediapipe opencv-python secure-smtplib

Cấu hình Gmail:
    1. Vào https://myaccount.google.com/apppasswords
    2. Tạo "App Password" cho ứng dụng (cần bật 2FA trước)
    3. Điền thông tin vào phần CONFIG bên dưới

Cử chỉ gửi mail: ✌️ Hòa Bình (V) - giữ 1.5 giây
"""

import cv2
import time
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime

# ── Thử API mới trước, fallback sang API cũ ──────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
    USE_NEW_API = True
except (ImportError, AttributeError):
    USE_NEW_API = False

import mediapipe as mp


# ══════════════════════════════════════════════════════════════════════════════
#  ⚙️  CẤU HÌNH GMAIL — Chỉnh sửa phần này
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "sender_email":    "kd126126@gmail.com",       # Gmail của bạn
    "sender_password": "djkp thwk dvjp vigf",        # App Password (16 ký tự)
    "receiver_email":  "quockhai2511@gmail.com",         # Email người nhận
    "subject":         "📸 Cử chỉ V được phát hiện!",
    "body":            "Xin chào!\n\nHệ thống nhận diện bàn tay vừa phát hiện cử chỉ Hòa Bình ✌️\n\nThời gian: {time}\n\nEmail này được gửi tự động.",
    "attach_screenshot": True,   # True = đính kèm ảnh chụp màn hình
    "hold_seconds":    1.5,      # Giữ cử chỉ bao lâu để gửi (giây)
    "cooldown_seconds": 10,      # Thời gian chờ giữa các lần gửi (giây)
}
# ══════════════════════════════════════════════════════════════════════════════


# ── Đếm ngón tay ─────────────────────────────────────────────────────────────
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

def count_fingers_from_list(landmarks, hand_label):
    count = 0
    if hand_label == "Right":
        if landmarks[4][0] < landmarks[3][0]: count += 1
    else:
        if landmarks[4][0] > landmarks[3][0]: count += 1
    for i in range(1, 5):
        if landmarks[FINGER_TIPS[i]][1] < landmarks[FINGER_PIPS[i]][1]:
            count += 1
    return count

GESTURES = {
    0: "Nam tay ✊",
    1: "Mot ngon ☝️",
    2: "Hoa binh ✌️  ← GUI MAIL",
    3: "Ba ngon 🤟",
    4: "Bon ngon 🖖",
    5: "Xoe tay 🖐️",
}

def get_gesture(n):
    return GESTURES.get(n, f"{n} ngon")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]


# ══════════════════════════════════════════════════════════════════════════════
#  📧 Gửi Gmail
# ══════════════════════════════════════════════════════════════════════════════
class GmailSender:
    def __init__(self):
        self.last_sent_time = 0
        self.is_sending = False
        self.status_message = ""
        self.status_color = (255, 255, 255)

    def send_email_async(self, screenshot=None):
        """Gửi email trong thread riêng để không block camera"""
        thread = threading.Thread(target=self._send_email, args=(screenshot,), daemon=True)
        thread.start()

    def _send_email(self, screenshot=None):
        self.is_sending = True
        self.status_message = "Dang gui email..."
        self.status_color = (0, 255, 255)

        try:
            msg = MIMEMultipart()
            msg["From"]    = CONFIG["sender_email"]
            msg["To"]      = CONFIG["receiver_email"]
            msg["Subject"] = CONFIG["subject"]

            body = CONFIG["body"].format(time=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            msg.attach(MIMEText(body, "plain", "utf-8"))

            # Đính kèm ảnh nếu có
            if screenshot is not None and CONFIG["attach_screenshot"]:
                _, img_encoded = cv2.imencode(".jpg", screenshot)
                img_bytes = img_encoded.tobytes()
                img_attachment = MIMEImage(img_bytes, name="gesture_v.jpg")
                img_attachment.add_header("Content-Disposition", "attachment", filename="gesture_v.jpg")
                msg.attach(img_attachment)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
                server.login(CONFIG["sender_email"], CONFIG["sender_password"])
                server.sendmail(CONFIG["sender_email"], CONFIG["receiver_email"], msg.as_string())

            self.last_sent_time = time.time()
            self.status_message = f"✓ Email da gui luc {datetime.now().strftime('%H:%M:%S')}"
            self.status_color = (0, 255, 0)
            print(f"[EMAIL] Gui thanh cong luc {datetime.now().strftime('%H:%M:%S')}")

        except smtplib.SMTPAuthenticationError:
            self.status_message = "✗ Loi xac thuc! Kiem tra App Password"
            self.status_color = (0, 0, 255)
            print("[EMAIL] Loi: Sai email/password. Dung App Password tu Google!")
        except smtplib.SMTPException as e:
            self.status_message = f"✗ Loi SMTP: {str(e)[:30]}"
            self.status_color = (0, 0, 255)
            print(f"[EMAIL] Loi SMTP: {e}")
        except Exception as e:
            self.status_message = f"✗ Loi: {str(e)[:30]}"
            self.status_color = (0, 0, 255)
            print(f"[EMAIL] Loi khac: {e}")
        finally:
            self.is_sending = False

    def can_send(self):
        """Kiểm tra có thể gửi không (cooldown)"""
        return (not self.is_sending and
                time.time() - self.last_sent_time >= CONFIG["cooldown_seconds"])

    def draw_status(self, frame):
        """Vẽ trạng thái email lên frame"""
        if self.status_message:
            h, w = frame.shape[:2]
            cv2.putText(frame, self.status_message, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.status_color, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  🕐 Theo dõi cử chỉ V để kích hoạt gửi mail
# ══════════════════════════════════════════════════════════════════════════════
class GestureTracker:
    def __init__(self):
        self.v_start_time = None
        self.triggered = False

    def update(self, has_v_gesture):
        now = time.time()
        if has_v_gesture:
            if self.v_start_time is None:
                self.v_start_time = now
            elapsed = now - self.v_start_time
            return elapsed  # Trả về thời gian đã giữ
        else:
            self.v_start_time = None
            self.triggered = False
            return 0.0

    def should_trigger(self, elapsed):
        if elapsed >= CONFIG["hold_seconds"] and not self.triggered:
            self.triggered = True
            return True
        return False

    def draw_progress(self, frame, elapsed):
        """Vẽ thanh tiến trình giữ cử chỉ V"""
        if elapsed <= 0:
            return
        h, w = frame.shape[:2]
        progress = min(elapsed / CONFIG["hold_seconds"], 1.0)
        bar_w = 300
        bar_h = 20
        x, y = (w - bar_w) // 2, h - 60

        # Nền
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
        # Tiến trình
        fill_color = (0, 255, 100) if progress < 1.0 else (0, 200, 255)
        cv2.rectangle(frame, (x, y), (x + int(bar_w * progress), y + bar_h), fill_color, -1)
        # Viền
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (200, 200, 200), 2)
        # Text
        label = f"Giu cu chi V... {elapsed:.1f}s / {CONFIG['hold_seconds']}s"
        cv2.putText(frame, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  📷 Vẽ UI + xử lý frame chung
# ══════════════════════════════════════════════════════════════════════════════
def draw_ui(frame, fps, hand_count, gmail_sender, cooldown_remaining=0):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (320, 90), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Ban tay: {hand_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Giu V {CONFIG['hold_seconds']}s de gui mail", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if cooldown_remaining > 0:
        cv2.putText(frame, f"Cho {cooldown_remaining:.0f}s gui tiep...", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 2)

    gmail_sender.draw_status(frame)


def process_hands(frame, landmarks_list, handedness_list, gmail_sender, gesture_tracker):
    """Xử lý landmarks, vẽ và kiểm tra cử chỉ V"""
    h, w = frame.shape[:2]
    has_v = False

    for i, hand_lm in enumerate(landmarks_list):
        raw_label = handedness_list[i]
        label = "Left" if raw_label == "Right" else "Right"
        pts  = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand_lm]
        norm = hand_lm
        color = (0, 220, 80) if label == "Right" else (255, 120, 0)

        for c in HAND_CONNECTIONS:
            cv2.line(frame, pts[c[0]], pts[c[1]], color, 2)
        for p in pts:
            cv2.circle(frame, p, 5, (255, 255, 255), -1)
            cv2.circle(frame, p, 3, color, -1)

        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
        x2, y2 = min(w, max(xs) + 20), min(h, max(ys) + 20)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        n = count_fingers_from_list(norm, label)
        if n == 2:
            has_v = True
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 200, 255), 3)  # Viền vàng khi V

        cv2.putText(frame, "Trai" if label == "Left" else "Phai",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, get_gesture(n),
                    (x1, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Cập nhật tracker và gửi mail nếu cần
    elapsed = gesture_tracker.update(has_v)
    gesture_tracker.draw_progress(frame, elapsed)

    cooldown_remaining = 0
    if not gmail_sender.can_send():
        remaining = CONFIG["cooldown_seconds"] - (time.time() - gmail_sender.last_sent_time)
        cooldown_remaining = max(0, remaining)

    if has_v and gesture_tracker.should_trigger(elapsed) and gmail_sender.can_send():
        print("[GESTURE] Cu chi V duoc giu du lau → Gui email!")
        screenshot = frame.copy() if CONFIG["attach_screenshot"] else None
        gmail_sender.send_email_async(screenshot)

    return cooldown_remaining


# ══════════════════════════════════════════════════════════════════════════════
#  API MỚI (mediapipe >= 0.10)
# ══════════════════════════════════════════════════════════════════════════════
def run_new_api(gmail_sender, gesture_tracker):
    import urllib.request, os
    MODEL_PATH = "hand_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("Dang tai model (~8MB), vui long cho...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Tai xong!")

    BaseOptions    = mp_python.BaseOptions
    HandLandmarker = mp_vision.HandLandmarker

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera san sang! Nhan Q de thoat, S de chup anh.")
    prev = time.time(); shot = 0

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_img)
            hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0

            landmarks_list = []
            handedness_list = []
            for i, hand_lm in enumerate(result.hand_landmarks or []):
                landmarks_list.append([(lm.x, lm.y) for lm in hand_lm])
                handedness_list.append(result.handedness[i][0].category_name)

            cooldown = process_hands(frame, landmarks_list, handedness_list, gmail_sender, gesture_tracker)

            fps = 1 / (time.time() - prev + 1e-9); prev = time.time()
            draw_ui(frame, fps, hand_count, gmail_sender, cooldown)
            cv2.imshow("Nhan Dien Ban Tay + Gmail", frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27): break
            if k == ord('s'):
                shot += 1; fn = f"hand_{shot}.jpg"
                cv2.imwrite(fn, frame); print(f"Luu: {fn}")

    cap.release(); cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  API CŨ (mediapipe < 0.10)
# ══════════════════════════════════════════════════════════════════════════════
def run_old_api(gmail_sender, gesture_tracker):
    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera san sang! Nhan Q de thoat, S de chup anh.")
    prev = time.time(); shot = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)
        rgb.flags.writeable = True
        hand_count = len(res.multi_hand_landmarks) if res.multi_hand_landmarks else 0

        landmarks_list = []
        handedness_list = []
        if res.multi_hand_landmarks:
            for i, hand_lm in enumerate(res.multi_hand_landmarks):
                landmarks_list.append([(lm.x, lm.y) for lm in hand_lm.landmark])
                handedness_list.append(res.multi_handedness[i].classification[0].label)
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

        cooldown = process_hands(frame, landmarks_list, handedness_list, gmail_sender, gesture_tracker)

        fps = 1 / (time.time() - prev + 1e-9); prev = time.time()
        draw_ui(frame, fps, hand_count, gmail_sender, cooldown)
        cv2.imshow("Nhan Dien Ban Tay + Gmail", frame)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27): break
        if k == ord('s'):
            shot += 1; fn = f"hand_{shot}.jpg"
            cv2.imwrite(fn, frame); print(f"Luu: {fn}")

    cap.release(); cv2.destroyAllWindows(); hands.close()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  NHAN DIEN BAN TAY + GUI GMAIL")
    print("=" * 55)

    # Kiểm tra cấu hình
    if "your_email" in CONFIG["sender_email"]:
        print("\n⚠️  CHUA CAU HINH GMAIL!")
        print("   Vui long mo file va dien vao phan CONFIG:")
        print("   - sender_email:    Gmail cua ban")
        print("   - sender_password: App Password (16 ky tu)")
        print("   - receiver_email:  Email nguoi nhan")
        print("\n   Cach lay App Password:")
        print("   1. Vao https://myaccount.google.com/apppasswords")
        print("   2. Chon 'Other' → dat ten bat ky → tao")
        print("   3. Copy 16 ky tu vao sender_password")
        print("\n   Nhan Enter de chay thu (khong gui duoc mail)...")
        input()

    gmail_sender    = GmailSender()
    gesture_tracker = GestureTracker()

    version = tuple(int(x) for x in mp.__version__.split(".")[:2])
    print(f"\nMediaPipe phien ban: {mp.__version__}")
    print(f"Giu cu chi V (✌️) trong {CONFIG['hold_seconds']}s de gui email\n")

    if version >= (0, 10) and USE_NEW_API:
        print("Dung API moi (Tasks API)")
        run_new_api(gmail_sender, gesture_tracker)
    else:
        print("Dung API cu (solutions)")
        run_old_api(gmail_sender, gesture_tracker)