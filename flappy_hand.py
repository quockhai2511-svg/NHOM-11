"""
Flappy Bird - Điều khiển bằng tay!
Cách chơi:
  - XÒE TAY (>= 3 ngón) hoặc NHẤC TAY LÊN  → chim vỗ cánh / bay lên
  - NẮM TAY / hạ tay xuống                  → chim rơi tự nhiên
  - Vẫn hỗ trợ SPACE / MŨI TÊN LÊN         → backup bàn phím

Cài đặt: pip install mediapipe opencv-python pygame
"""

import pygame, random, time, threading, cv2
import mediapipe as mp
from pygame.locals import *

# ═══════════════════════════════════════════════════════════════
#  CẤU HÌNH GAME
# ═══════════════════════════════════════════════════════════════
SCREEN_W   = 400
SCREEN_H   = 600
SPEED      = 20
GRAVITY    = 2.5
GAME_SPEED = 15

GROUND_W   = 2 * SCREEN_W
GROUND_H   = 100
PIPE_W     = 80
PIPE_H     = 500
PIPE_GAP   = 150

wing = 'assets/audio/wing.wav'
hit  = 'assets/audio/hit.wav'

pygame.mixer.init()

# ═══════════════════════════════════════════════════════════════
#  HAND DETECTOR — chạy trên thread riêng
# ═══════════════════════════════════════════════════════════════
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

def _count_fingers(landmarks, hand_label):
    count = 0
    if hand_label == "Right":
        if landmarks[4][0] < landmarks[3][0]: count += 1
    else:
        if landmarks[4][0] > landmarks[3][0]: count += 1
    for i in range(1, 5):
        if landmarks[FINGER_TIPS[i]][1] < landmarks[FINGER_PIPS[i]][1]:
            count += 1
    return count

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

class HandController:
    """
    Chạy camera + MediaPipe trên thread nền.
    Thuộc tính công khai:
      .flap      → True nếu tay đang xòe (>= 3 ngón) hoặc nhấc lên cao
      .hand_img  → frame camera gần nhất (numpy array BGR)
    """
    def __init__(self):
        self.flap      = False
        self.hand_img  = None
        self._stop     = False
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True

    # ── Kiểm tra API mediapipe ────────────────────────────────
    def _run(self):
        version = tuple(int(x) for x in mp.__version__.split(".")[:2])
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
            if version >= (0, 10):
                self._run_new(mp_python, mp_vision, HandLandmarkerOptions, RunningMode)
                return
        except (ImportError, AttributeError):
            pass
        self._run_old()

    # ── Phát hiện vỗ tay: tay nhấc lên so với frame trước ────
    _prev_wrist_y = None
    FLAP_FINGERS  = 3   # >= N ngón = xòe tay → flap

    def _process(self, norm, hand_label, wrist_y_norm):
        n = _count_fingers(norm, hand_label)
        # Cách 1: xòe tay
        fingers_open = n >= self.FLAP_FINGERS
        # Cách 2: nhấc tay lên nhanh
        lift = False
        if self._prev_wrist_y is not None:
            lift = (self._prev_wrist_y - wrist_y_norm) > 0.04  # di chuyển lên > 4% chiều cao
        self._prev_wrist_y = wrist_y_norm
        self.flap = fingers_open or lift
        return n

    # ── API MỚI ───────────────────────────────────────────────
    def _run_new(self, mp_python, mp_vision, Opts, RunMode):
        import urllib.request, os
        MODEL = "hand_landmarker.task"
        if not os.path.exists(MODEL):
            print("[HandCtrl] Dang tai model hand_landmarker.task (~8MB)...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task", MODEL)
            print("[HandCtrl] Tai xong!")

        options = Opts(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=RunMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.6, min_tracking_confidence=0.5
        )
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        with mp_vision.HandLandmarker.create_from_options(options) as det:
            while not self._stop:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = det.detect(mp_img)

                detected = False
                for i, hand_lm in enumerate(res.hand_landmarks or []):
                    raw = res.handedness[i][0].category_name
                    label = "Left" if raw == "Right" else "Right"
                    norm  = [(lm.x, lm.y) for lm in hand_lm]
                    pts   = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lm]
                    color = (0, 220, 80) if label == "Right" else (255, 120, 0)

                    n = self._process(norm, label, norm[0][1])
                    detected = True

                    for c in HAND_CONNECTIONS:
                        cv2.line(frame, pts[c[0]], pts[c[1]], color, 2)
                    for p in pts:
                        cv2.circle(frame, p, 4, (255,255,255), -1)
                        cv2.circle(frame, p, 2, color, -1)

                    status = "FLAP!" if self.flap else f"{n} ngon"
                    clr2 = (0,255,0) if self.flap else (200,200,200)
                    cv2.putText(frame, status, (pts[0][0], pts[0][1]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr2, 2)

                if not detected:
                    self.flap = False
                    self._prev_wrist_y = None

                # Hiển thị gợi ý
                cv2.putText(frame, "Xoe tay = FLAP", (10, h-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,100), 1)
                self.hand_img = frame

        cap.release()

    # ── API CŨ ────────────────────────────────────────────────
    def _run_old(self):
        mp_hands  = mp.solutions.hands
        mp_draw   = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                               min_detection_confidence=0.7, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while not self._stop:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)
            rgb.flags.writeable = True

            detected = False
            if res.multi_hand_landmarks:
                for i, hand_lm in enumerate(res.multi_hand_landmarks):
                    raw   = res.multi_handedness[i].classification[0].label
                    label = "Left" if raw == "Right" else "Right"
                    norm  = [(lm.x, lm.y) for lm in hand_lm.landmark]

                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())

                    n = self._process(norm, label, norm[0][1])
                    detected = True

                    x1 = max(0, int(min(lm.x*w for lm in hand_lm.landmark)) - 10)
                    y1 = max(0, int(min(lm.y*h for lm in hand_lm.landmark)) - 10)
                    status = "FLAP!" if self.flap else f"{n} ngon"
                    clr = (0,255,0) if self.flap else (200,200,200)
                    cv2.putText(frame, status, (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2)

            if not detected:
                self.flap = False
                self._prev_wrist_y = None

            cv2.putText(frame, "Xoe tay = FLAP", (10, h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,100), 1)
            self.hand_img = frame

        cap.release()
        hands.close()


# ═══════════════════════════════════════════════════════════════
#  GAME SPRITES
# ═══════════════════════════════════════════════════════════════
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha(),
        ]
        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[0]
        self.mask  = pygame.mask.from_surface(self.image)
        self.rect  = self.image.get_rect()
        self.rect[0] = SCREEN_W / 6
        self.rect[1] = SCREEN_H / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]


class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_W, PIPE_H))
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_H - ysize
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_W, GROUND_H))
        self.mask  = pygame.mask.from_surface(self.image)
        self.rect  = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_H - GROUND_H

    def update(self):
        self.rect[0] -= GAME_SPEED


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    return Pipe(False, xpos, size), Pipe(True, xpos, SCREEN_H - size - PIPE_GAP)

def show_cam(surface, hand_img, x=10, y=10, w=200, h=150):
    """Vẽ ảnh camera nhỏ góc trên trái màn hình game."""
    if hand_img is None:
        return
    rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (w, h))
    pg_surf = pygame.surfarray.make_surface(small.swapaxes(0, 1))
    # Viền
    pygame.draw.rect(surface, (255,255,0), (x-2, y-2, w+4, h+4), 2)
    surface.blit(pg_surf, (x, y))


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption('Flappy Bird - Hand Control')

    BG    = pygame.transform.scale(
                pygame.image.load('assets/sprites/background-day.png'), (SCREEN_W, SCREEN_H))
    MSG   = pygame.image.load('assets/sprites/message.png').convert_alpha()
    font  = pygame.font.SysFont('Arial', 18, bold=True)

    # Khởi động hand controller
    print("[Game] Khoi dong camera nhan dien tay...")
    ctrl = HandController()
    time.sleep(1.5)   # Chờ camera warm-up

    # Sprites
    bird_group  = pygame.sprite.Group()
    bird        = Bird()
    bird_group.add(bird)

    ground_group = pygame.sprite.Group()
    for i in range(2):
        ground_group.add(Ground(GROUND_W * i))

    pipe_group = pygame.sprite.Group()
    for i in range(2):
        p = get_random_pipes(SCREEN_W * i + 800)
        pipe_group.add(p[0], p[1])

    clock = pygame.time.Clock()
    score = 0

    # ── MÀN HÌNH BẮT ĐẦU ───────────────────────────────────────
    prev_flap = False
    begin = True
    while begin:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == QUIT:
                ctrl.stop(); pygame.quit(); return
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP):
                bird.bump()
                try: pygame.mixer.music.load(wing); pygame.mixer.music.play()
                except: pass
                begin = False

        # Xòe tay để bắt đầu
        cur_flap = ctrl.flap
        if cur_flap and not prev_flap:
            bird.bump()
            try: pygame.mixer.music.load(wing); pygame.mixer.music.play()
            except: pass
            begin = False
        prev_flap = cur_flap

        screen.blit(BG, (0, 0))
        screen.blit(MSG, (120, 150))

        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_W - 20))

        bird.begin()
        ground_group.update()
        bird_group.draw(screen)
        ground_group.draw(screen)
        show_cam(screen, ctrl.hand_img)

        hint = font.render("Xoe tay de bat dau!", True, (255, 230, 0))
        screen.blit(hint, (SCREEN_W//2 - hint.get_width()//2, SCREEN_H - 40))
        pygame.display.update()

    # ── VÒNG GAME CHÍNH ────────────────────────────────────────
    prev_flap = False
    while True:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == QUIT:
                ctrl.stop(); pygame.quit(); return
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP):
                bird.bump()
                try: pygame.mixer.music.load(wing); pygame.mixer.music.play()
                except: pass

        # Điều khiển tay (edge-trigger: chỉ bump khi vừa xòe, không giữ liên tục)
        cur_flap = ctrl.flap
        if cur_flap and not prev_flap:
            bird.bump()
            try: pygame.mixer.music.load(wing); pygame.mixer.music.play()
            except: pass
        prev_flap = cur_flap

        screen.blit(BG, (0, 0))

        # Cuộn đất
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_W - 20))

        # Thêm ống mới
        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])
            p = get_random_pipes(SCREEN_W * 2)
            pipe_group.add(p[0], p[1])
            score += 1

        bird_group.update()
        ground_group.update()
        pipe_group.update()

        pipe_group.draw(screen)
        ground_group.draw(screen)
        bird_group.draw(screen)
        show_cam(screen, ctrl.hand_img)

        # Điểm số
        score_surf = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_surf, (SCREEN_W - score_surf.get_width() - 10, 10))

        # Trạng thái tay
        status_txt = "FLAP!" if ctrl.flap else "..."
        color_txt  = (0, 255, 80) if ctrl.flap else (200, 200, 200)
        st_surf = font.render(status_txt, True, color_txt)
        screen.blit(st_surf, (10, 170))

        pygame.display.update()

        # Va chạm
        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
            try: pygame.mixer.music.load(hit); pygame.mixer.music.play()
            except: pass
            time.sleep(1)
            break

    # Game over
    go_font = pygame.font.SysFont('Arial', 36, bold=True)
    go_surf = go_font.render("GAME OVER", True, (255, 50, 50))
    sc_surf = font.render(f"Diem: {score}  |  Nhan SPACE de choi lai", True, (255,255,255))
    screen.blit(go_surf, (SCREEN_W//2 - go_surf.get_width()//2, SCREEN_H//2 - 40))
    screen.blit(sc_surf, (SCREEN_W//2 - sc_surf.get_width()//2, SCREEN_H//2 + 10))
    pygame.display.update()
    time.sleep(2)

    ctrl.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
