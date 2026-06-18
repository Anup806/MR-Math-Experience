import cv2
import mediapipe as mp
import random
import time
import numpy as np
import math
import sys
import csv
import os
import json
import pickle
from datetime import datetime
from urllib.request import urlretrieve
from types import SimpleNamespace

try:
    import pygame
except Exception:
    print("Error: pygame is not installed. Install with: python -m pip install pygame")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, Input
    DKT_AVAILABLE = True
except Exception:
    DKT_AVAILABLE = False

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

HAND_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
hand_landmarker = None


def load_hand_landmarker():
    global hand_landmarker
    if hand_landmarker is not None:
        return hand_landmarker

    try:
        if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
            print("Downloading hand landmark model...")
            urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)

        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
        print("✓ Hand gesture tracking initialized")
    except Exception as e:
        print(f"⚠ Hand gesture tracking unavailable: {e}")
        hand_landmarker = None

    return hand_landmarker


def detect_hand_landmarks(frame_bgr):
    if hand_landmarker is None:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    result = hand_landmarker.detect(mp_image)
    if result and result.hand_landmarks:
        return SimpleNamespace(landmark=result.hand_landmarks[0])
    return None

# ─────────────────────────────────────────────────────────────
# PC-BKT  MODEL  INTEGRATION
# ─────────────────────────────────────────────────────────────
BKT_MODEL_PATH = "pc_bkt_model__1_.pkl"
bkt_model = None

try:
    with open(BKT_MODEL_PATH, "rb") as f:
        bkt_model = pickle.load(f)
    print(f"✓ PC-BKT KMeans model loaded  (clusters={bkt_model.n_clusters})")
    # Sort cluster indices by p_know (feature index 2) low→high
    _centers = bkt_model.cluster_centers_
    _order   = np.argsort(_centers[:, 2])          # [BASIC, INTER, ADVANCED] cluster ids
    CLUSTER_TO_LEVEL = {int(_order[0]): "BASIC",
                        int(_order[1]): "INTERMEDIATE",
                        int(_order[2]): "ADVANCED"}
    print(f"  Cluster→Level map: {CLUSTER_TO_LEVEL}")
except Exception as e:
    print(f"⚠  Could not load BKT model ({e}). Falling back to threshold-based classification.")
    bkt_model = None
    CLUSTER_TO_LEVEL = {}

# ─── BKT  PARAMETERS  (Bayesian Knowledge Tracing) ──────────
BKT_P_INIT    = 0.3    # Prior P(knows skill at start)
BKT_P_LEARN   = 0.2    # P(learns on each attempt)
BKT_P_FORGET  = 0.05   # P(forgets after knowing)
BKT_P_SLIP    = 0.1    # P(wrong answer even if knows)
BKT_P_GUESS   = 0.25   # P(right answer even if doesn't know)

# Difficulty thresholds used when model is unavailable
MASTERY_BASIC        = 0.40
MASTERY_INTERMEDIATE = 0.70

# Numbers per difficulty level
LEVEL_NUMBERS = {
    "BASIC":        [1, 2, 3],
    "INTERMEDIATE": [4, 5, 6],
    "ADVANCED":     [7, 8, 9, 10],
}

# Assessment phase
ASSESSMENT_QUESTIONS = 15          # Fixed first-session count
ASSESSMENT_NUMBERS   = [1, 2, 3]  # 5 of each
ASSESSMENT_PER_NUM   = 5

# Adaptive re-evaluation cadence (session 2+)
ADAPTIVE_REEVAL_EVERY = 5

# ─── STUDENT  LEVEL  PERSISTENCE ────────────────────────────
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR         = os.path.join(SCRIPT_DIR, "photos")
AUDIO_DIR         = os.path.join(SCRIPT_DIR, "audio")
CSV_DIR           = os.path.join(SCRIPT_DIR, "game_data")
INTERACTIONS_CSV  = os.path.join(CSV_DIR, "interactions.csv")
SESSIONS_CSV      = os.path.join(CSV_DIR, "sessions.csv")
STUDENT_LEVELS    = os.path.join(CSV_DIR, "student_levels.json")

INTERACTIONS_HEADERS = [
    'student_name','age','student_grade','game_mode','timestamp',
    'math_problem','user_answer','correct_answer','reaction_time_s',
    'correct','score','session_id','total_screen_time',
    'p_know','difficulty_level',
]
SESSIONS_HEADERS = [
    'student_name','age','student_grade','game_mode','session_start',
    'session_end','final_score','total_problems','correct_problems',
    'accuracy','session_id','total_screen_time','average_reaction_time',
    'final_p_know','final_level','session_number',
]

# ─────────────────────────────────────────────────────────────
# BKT  ENGINE
# ─────────────────────────────────────────────────────────────
class BKTEngine:
    """Bayesian Knowledge Tracing with KMeans difficulty classification."""

    def __init__(self):
        self.p_know      = BKT_P_INIT
        self.history     = []        # list of (correct: bool, reaction_time: float)
        self.answer_count = 0

    def update(self, correct: bool, reaction_time: float):
        """Update BKT posterior after one answer."""
        self.history.append((correct, reaction_time))
        self.answer_count += 1

        # P(knows | evidence)  via Bayes
        if correct:
            p_evidence_given_know    = 1.0 - BKT_P_SLIP
            p_evidence_given_notknow = BKT_P_GUESS
        else:
            p_evidence_given_know    = BKT_P_SLIP
            p_evidence_given_notknow = 1.0 - BKT_P_GUESS

        p_know_and_evidence    = self.p_know * p_evidence_given_know
        p_notknow_and_evidence = (1.0 - self.p_know) * p_evidence_given_notknow
        denom = p_know_and_evidence + p_notknow_and_evidence
        if denom > 0:
            self.p_know = p_know_and_evidence / denom

        # Learning opportunity (forget is very small)
        self.p_know = self.p_know * (1 - BKT_P_FORGET) + (1 - self.p_know) * BKT_P_LEARN
        self.p_know = max(0.0, min(1.0, self.p_know))

    # ── Feature vector for KMeans prediction ─────────────────
    def compute_features(self, window: int = 10) -> np.ndarray:
        recent = self.history[-window:] if len(self.history) >= window else self.history
        if not recent:
            return np.array([[0.5, 0.5, self.p_know, 0.5, 0.5]])

        # feat[0]  accuracy_rate
        accuracy_rate = sum(1 for c, _ in recent if c) / len(recent)

        # feat[1]  speed_score  (1 – norm. avg RT, clipped 0–30s)
        rts = [rt for _, rt in recent]
        avg_rt = sum(rts) / len(rts)
        speed_score = max(0.0, 1.0 - avg_rt / 30.0)

        # feat[2]  p_know
        p_know = self.p_know

        # feat[3]  consistency_score
        flags = [1.0 if c else 0.0 for c, _ in recent]
        std   = float(np.std(flags)) if len(flags) > 1 else 0.0
        consistency_score = max(0.0, 1.0 - std)

        # feat[4]  learning_trend
        if len(flags) >= 4:
            half = len(flags) // 2
            first_half  = sum(flags[:half])  / half
            second_half = sum(flags[half:])  / (len(flags) - half)
            learning_trend = max(0.0, min(1.0, 0.5 + (second_half - first_half)))
        else:
            learning_trend = 0.5

        return np.array([[accuracy_rate, speed_score, p_know, consistency_score, learning_trend]])

    def classify_level(self) -> str:
        """Classify student level using KMeans model or threshold fallback."""
        if bkt_model is not None:
            try:
                feats   = self.compute_features()
                cluster = int(bkt_model.predict(feats)[0])
                level   = CLUSTER_TO_LEVEL.get(cluster, "BASIC")
                print(f"  BKT KMeans → cluster={cluster}, level={level}, p_know={self.p_know:.3f}")
                return level
            except Exception as e:
                print(f"  KMeans predict error: {e}, using threshold fallback")

        # Threshold fallback
        if self.p_know < MASTERY_BASIC:
            return "BASIC"
        elif self.p_know < MASTERY_INTERMEDIATE:
            return "INTERMEDIATE"
        else:
            return "ADVANCED"


# ─────────────────────────────────────────────────────────────
# QUESTION  QUEUE  MANAGER
# ─────────────────────────────────────────────────────────────
class QuestionManager:
    """
    Session 1:  Assessment phase – 15 fixed questions (5 × {1,2,3}).
    Session 2+: Adaptive phase   – questions drawn from level-appropriate pool.
    """

    def __init__(self, player_key: str, session_number: int, initial_level: str = "BASIC"):
        self.player_key     = player_key
        self.session_number = session_number
        self.current_level  = initial_level
        self.queue          = []
        self.is_assessment  = (session_number == 1)
        self.questions_done = 0

        if self.is_assessment:
            self._build_assessment_queue()
        else:
            self._build_adaptive_queue()

    def _build_assessment_queue(self):
        q = []
        for n in ASSESSMENT_NUMBERS:
            q.extend([n] * ASSESSMENT_PER_NUM)
        random.shuffle(q)
        self.queue = q
        print(f"  Assessment queue: {self.queue}")

    def _build_adaptive_queue(self, batch: int = 10):
        pool = LEVEL_NUMBERS.get(self.current_level, [1, 2, 3])
        # Distribute evenly across the pool
        q = []
        per_num = max(1, batch // len(pool))
        for n in pool:
            q.extend([n] * per_num)
        random.shuffle(q)
        self.queue = q
        print(f"  Adaptive queue ({self.current_level}): {self.queue}")

    def next_number(self) -> int:
        if not self.queue:
            if self.is_assessment:
                # Assessment done – re-fill is not expected but guard anyway
                self._build_assessment_queue()
            else:
                self._build_adaptive_queue()
        return self.queue.pop(0)

    def assessment_complete(self) -> bool:
        return self.is_assessment and self.questions_done >= ASSESSMENT_QUESTIONS

    def update_level(self, new_level: str):
        if new_level != self.current_level:
            print(f"  Level updated: {self.current_level} → {new_level}")
            self.current_level = new_level
            # Rebuild queue with new level
            self._build_adaptive_queue()
        # Transition out of assessment mode when level is set
        self.is_assessment = False

    def record_answer(self):
        self.questions_done += 1


# ─────────────────────────────────────────────────────────────
# STUDENT  LEVEL  PERSISTENCE
# ─────────────────────────────────────────────────────────────
def load_student_data(player_key: str) -> dict:
    try:
        if os.path.exists(STUDENT_LEVELS):
            with open(STUDENT_LEVELS, 'r') as f:
                data = json.load(f)
            return data.get(player_key, {})
    except Exception as e:
        print(f"Error loading student data: {e}")
    return {}

def save_student_data(player_key: str, level: str, p_know: float, session_number: int):
    try:
        data = {}
        if os.path.exists(STUDENT_LEVELS):
            with open(STUDENT_LEVELS, 'r') as f:
                data = json.load(f)
        data[player_key] = {
            "level": level,
            "p_know": p_know,
            "session_number": session_number,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(STUDENT_LEVELS, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {player_key} → level={level}, p_know={p_know:.3f}, session={session_number}")
    except Exception as e:
        print(f"Error saving student data: {e}")


# ─────────────────────────────────────────────────────────────
# CSV  LOGGING
# ─────────────────────────────────────────────────────────────
def initialize_csv_files():
    try:
        os.makedirs(CSV_DIR, exist_ok=True)
        if not os.path.exists(INTERACTIONS_CSV):
            with open(INTERACTIONS_CSV, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(INTERACTIONS_HEADERS)
        if not os.path.exists(SESSIONS_CSV):
            with open(SESSIONS_CSV, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(SESSIONS_HEADERS)
        print(f"✓ CSV files in: {CSV_DIR}")
    except Exception as e:
        print(f"CSV init error: {e}")

def log_interaction(player_name, age, student_grade, game_mode, math_problem,
                    user_answer, correct_answer, reaction_time_s, correct,
                    current_score, session_id, total_screen_time, p_know, level):
    try:
        with open(INTERACTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                player_name, age, student_grade, game_mode,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                math_problem, user_answer, correct_answer,
                f"{reaction_time_s:.3f}" if reaction_time_s is not None else "",
                1 if correct else 0, current_score, session_id,
                f"{total_screen_time:.2f}", f"{p_know:.4f}", level,
            ])
    except Exception as e:
        print(f"Interaction log error: {e}")

def log_session_end(player_name, age, student_grade, game_mode, session_start,
                    final_score, total_problems, correct_problems, session_id,
                    total_screen_time, average_reaction_time,
                    final_p_know, final_level, session_number):
    try:
        accuracy = (correct_problems / total_problems * 100) if total_problems > 0 else 0
        with open(SESSIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                player_name, age, student_grade, game_mode,
                session_start.strftime('%Y-%m-%d %H:%M:%S'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                final_score, total_problems, correct_problems,
                f"{accuracy:.2f}", session_id,
                f"{total_screen_time:.2f}",
                f"{average_reaction_time:.3f}" if average_reaction_time is not None else "",
                f"{final_p_know:.4f}", final_level, session_number,
            ])
    except Exception as e:
        print(f"Session log error: {e}")


# ─────────────────────────────────────────────────────────────
# PYGAME  SETUP
# ─────────────────────────────────────────────────────────────
pygame.init()
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
    print("✓ Audio initialized")
except Exception as e:
    print(f"⚠ Audio init failed: {e}")

try:
    WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[0]
except Exception:
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption("Math Learning – BKT Adaptive AR")
pygame.event.set_blocked(pygame.MOUSEMOTION)
pygame.mouse.set_visible(True)

font        = pygame.font.Font(None, 60)
large_font  = pygame.font.Font(None, 100)
small_font  = pygame.font.Font(None, 36)
tiny_font   = pygame.font.Font(None, 24)
number_font = pygame.font.Font(None, 300)
clock       = pygame.time.Clock()

# ─── MediaPipe ───────────────────────────────────────────────
load_hand_landmarker()

# ─── Camera ──────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

initialize_csv_files()

# ─── Geometry ────────────────────────────────────────────────
APPLE_SIZE = 300
SPAWN_X, SPAWN_Y = 120, 300
basket = {"x": (WIDTH - 600) // 2, "y": HEIGHT - 460, "w": 600, "h": 400}
submit = {"x": WIDTH - 380,        "y": HEIGHT - 380, "w": 220, "h": 90}

GAME_MODES = ["COUNTING", "ADDITION"]
PINCH_THRESHOLD = 45
PINCH_DELAY     = 0.3

# ─── Sounds ──────────────────────────────────────────────────
audio_loaded     = False
congrats_sound   = None
tryagain_sound   = None
counting_sounds  = {}
number_pronunciation = {}

def try_load_sound(base, exts=('.MP3', '.WAV', '.OGG', '.mp3', '.wav')):
    for ext in exts:
        try:
            filename = os.path.join(AUDIO_DIR, base + ext)
            s = pygame.mixer.Sound(filename)
            print(f"  ✓ {filename}")
            return s
        except Exception:
            pass
    return None

congrats_sound = try_load_sound("Cheer") or try_load_sound("congratulation")
tryagain_sound = try_load_sound("tryagain")
for i in range(1, 11):
    counting_sounds[i] = try_load_sound(str(i))
    if counting_sounds[i]:
        counting_sounds[i].set_volume(0.6)
for i in range(1, 11):
    number_pronunciation[i] = try_load_sound(f"number_{i}") or counting_sounds.get(i)

sounds_loaded = sum(1 for s in counting_sounds.values() if s)
audio_loaded  = sounds_loaded > 0 or bool(congrats_sound) or bool(tryagain_sound)
print(f"  Audio ready: {sounds_loaded} counting sounds loaded")

def play_sound(sound, allow_interrupt=True, volume=None):
    if not audio_loaded or sound is None:
        return False
    try:
        orig = sound.get_volume()
        if volume is not None:
            sound.set_volume(volume)
        if allow_interrupt:
            sound.stop(); sound.play()
        elif not pygame.mixer.get_busy():
            sound.play()
        if volume is not None:
            sound.set_volume(orig)
        return True
    except Exception:
        return False

# ─── Apple / Basket images ───────────────────────────────────
try:
    apple_surface = pygame.image.load(os.path.join(PHOTO_DIR, "apple.png")).convert_alpha()
    apple_surface = pygame.transform.scale(apple_surface, (APPLE_SIZE, APPLE_SIZE))
except Exception:
    apple_surface = pygame.Surface((APPLE_SIZE, APPLE_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(apple_surface, (255, 50, 50),
                       (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 10)
    pygame.draw.rect(apple_surface, (100, 70, 20),
                     pygame.Rect(APPLE_SIZE//2-5, APPLE_SIZE//4-10, 10, APPLE_SIZE//4))
    pygame.draw.ellipse(apple_surface, (100, 200, 50),
                        pygame.Rect(APPLE_SIZE//2+5, APPLE_SIZE//4-15, 15, 10))

small_apple_surface = pygame.transform.scale(apple_surface, (80, 80))
empty_apple_surface = small_apple_surface.copy()
empty_apple_surface.fill((255, 255, 255, 100), special_flags=pygame.BLEND_RGBA_MULT)

basket_images = []
for i in range(11):
    try:
        img = pygame.image.load(os.path.join(PHOTO_DIR, f"basket{i}.png")).convert_alpha()
        basket_images.append(pygame.transform.scale(img, (basket["w"], basket["h"])))
    except Exception:
        surf = pygame.Surface((basket["w"], basket["h"]), pygame.SRCALPHA)
        pygame.draw.ellipse(surf, (210, 180, 140),
                            (10, basket["h"]//2, basket["w"]-20, basket["h"]//2))
        pygame.draw.arc(surf, (160, 120, 80),
                        (basket["w"]//4, 10, basket["w"]//2, 50), math.pi, 2*math.pi, 5)
        basket_images.append(surf)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def inside(x, y, rect):
    return rect["x"] < x < rect["x"]+rect["w"] and rect["y"] < y < rect["y"]+rect["h"]

def compute_pinch_state(lms, w, h):
    t, i = lms.landmark[4], lms.landmark[8]
    return math.hypot((t.x-i.x)*w, (t.y-i.y)*h) < PINCH_THRESHOLD

def get_finger_pos(lms, w, h):
    t, i = lms.landmark[4], lms.landmark[8]
    return (int(t.x*w), int(t.y*h)), (int(i.x*w), int(i.y*h))

def draw_hand_skeleton(screen, lms, w, h):
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
             (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
    for s, e in conns:
        ps = (int(lms.landmark[s].x*w), int(lms.landmark[s].y*h))
        pe = (int(lms.landmark[e].x*w), int(lms.landmark[e].y*h))
        pygame.draw.line(screen, (0, 255, 0), ps, pe, 2)
    for lm in lms.landmark:
        pygame.draw.circle(screen, (255,0,0), (int(lm.x*w), int(lm.y*h)), 4)
    tp, ip = get_finger_pos(lms, w, h)
    pygame.draw.line(screen, (255, 255, 0), tp, ip, 2)

def cvimage_to_pygame(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "RGB")
    except Exception:
        return None

def display_camera_fullscreen(screen, img):
    if img is not None:
        try:
            surf = cvimage_to_pygame(cv2.resize(img, (WIDTH, HEIGHT)))
            if surf:
                screen.blit(surf, (0, 0))
                ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                ov.fill((0, 0, 0, 40))
                screen.blit(ov, (0, 0))
                return
        except Exception:
            pass
    screen.fill((0, 0, 0))

def spawn_apple():
    return {"x": SPAWN_X, "y": SPAWN_Y, "picked": False, "in_basket": False}

def reset_game(apples_ref, dropped_ref):
    return [spawn_apple()], 0


# ─── LEVEL COLORS ────────────────────────────────────────────
LEVEL_COLORS = {
    "BASIC":        (100, 200, 100),   # green
    "INTERMEDIATE": (255, 200,  50),   # yellow
    "ADVANCED":     (255, 100,  50),   # orange-red
}
LEVEL_LABELS = {
    "BASIC":        "BASIC",
    "INTERMEDIATE": "INTERMEDIATE",
    "ADVANCED":     "ADVANCED",
}


# ─────────────────────────────────────────────────────────────
# SCREEN 1 – PLAYER  INPUT
# ─────────────────────────────────────────────────────────────
player_name    = ""
player_age     = ""
student_grade  = ""
input_focus    = "name"
name_entry_active = True
running = False

while name_entry_active:
    clock.tick(30)
    screen.fill((30, 30, 40))

    t = large_font.render("Math Learning Game", True, (100, 200, 255))
    screen.blit(t, ((WIDTH-t.get_width())//2, HEIGHT//6))
    t = font.render("Enter your details:", True, (255,255,255))
    screen.blit(t, ((WIDTH-t.get_width())//2, HEIGHT//4))

    bw, bh = 500, 60
    bx = (WIDTH-bw)//2
    sy = HEIGHT//2 - 120

    for label, val, yo, field in [
        ("Name:",        player_name,   0,   "name"),
        ("Age:",         player_age,  110,   "age"),
        ("Grade/Level:", student_grade, 220, "grade"),
    ]:
        lbl = small_font.render(label, True, (200,200,200))
        screen.blit(lbl, (bx, sy+yo))
        pygame.draw.rect(screen, (50,50,60),   (bx, sy+yo+30, bw, bh))
        col = (100,150,255) if input_focus==field else (80,120,200)
        pygame.draw.rect(screen, col,          (bx, sy+yo+30, bw, bh), 3)
        disp = val if val else "_"
        screen.blit(font.render(disp, True, (255,255,255)), (bx+20, sy+yo+40))

    hint = small_font.render("TAB to switch fields  |  ENTER to continue", True, (150,200,150))
    screen.blit(hint, ((WIDTH-hint.get_width())//2, HEIGHT-100))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            name_entry_active = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if player_name.strip() and player_age.strip() and student_grade.strip():
                    name_entry_active = False
                    running = True
            elif event.key == pygame.K_BACKSPACE:
                if input_focus == "name":    player_name    = player_name[:-1]
                elif input_focus == "age":   player_age     = player_age[:-1]
                elif input_focus == "grade": student_grade  = student_grade[:-1]
            elif event.key == pygame.K_TAB:
                input_focus = {"name":"age","age":"grade","grade":"name"}[input_focus]
            elif event.unicode:
                if input_focus == "name"  and len(player_name)   < 30 and event.unicode.isprintable():
                    player_name   += event.unicode
                elif input_focus == "age" and len(player_age)    < 3  and event.unicode.isdigit():
                    player_age    += event.unicode
                elif input_focus == "grade" and len(student_grade) < 50 and event.unicode.isprintable():
                    student_grade += event.unicode

if not running:
    cap.release(); pygame.quit(); sys.exit()

# ─── Load/init student profile ───────────────────────────────
player_key = f"{player_name.strip().lower()}_{player_age.strip()}"
student_data = load_student_data(player_key)
session_number = student_data.get("session_number", 0) + 1
saved_level    = student_data.get("level", "BASIC")
saved_p_know   = student_data.get("p_know", BKT_P_INIT)

print(f"\nPlayer: {player_name}  |  Session #{session_number}")
print(f"  Saved level: {saved_level}  |  Saved p_know: {saved_p_know:.3f}")

is_assessment_session = (session_number == 1)

# Initialise BKT
bkt = BKTEngine()
bkt.p_know = saved_p_know if not is_assessment_session else BKT_P_INIT

# Initialise question manager
current_level = "BASIC" if is_assessment_session else saved_level
q_manager = QuestionManager(player_key, session_number, current_level)


# ─────────────────────────────────────────────────────────────
# SCREEN 2 – GAME  MODE  SELECTION
# ─────────────────────────────────────────────────────────────
GAME_MODE = None
game_sel_active = True

mode_buttons = []
bw2, bh2 = 400, 150
bsp       = 100
tot_w     = bw2*2 + bsp
sx2       = (WIDTH - tot_w)//2
yp2       = HEIGHT//2 - bh2//2

for i, mode in enumerate(GAME_MODES):
    mode_buttons.append({
        "mode": mode,
        "rect": pygame.Rect(sx2 + i*(bw2+bsp), yp2, bw2, bh2),
        "color":       (70,130,180) if mode=="COUNTING" else (60,179,113),
        "hover_color": (100,160,210) if mode=="COUNTING" else (85,199,133),
    })

while game_sel_active and running:
    clock.tick(30)
    ok, img = cap.read()
    if not ok: img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    display_camera_fullscreen(screen, img)

    finger_pos      = None
    current_pinch   = False
    lms = None
    if hand_landmarker is not None:
        try:
            fr = cv2.flip(cv2.resize(img, (WIDTH, HEIGHT)), 1)
            lms = detect_hand_landmarks(fr)
        except Exception:
            lms = None

    if lms:
        finger_pos   = (int(lms.landmark[8].x*WIDTH), int(lms.landmark[8].y*HEIGHT))
        current_pinch = compute_pinch_state(lms, WIDTH, HEIGHT)
        draw_hand_skeleton(screen, lms, WIDTH, HEIGHT)
    else:
        finger_pos = pygame.mouse.get_pos()
        current_pinch = bool(pygame.mouse.get_pressed(3)[0])

    bg = pygame.Surface((WIDTH,120), pygame.SRCALPHA)
    bg.fill((0,0,0,150)); screen.blit(bg,(0,50))
    t = large_font.render("SELECT GAME MODE", True, (255,255,200))
    screen.blit(t, ((WIDTH-t.get_width())//2, 80))

    # Session info banner
    if is_assessment_session:
        banner = font.render(f"SESSION 1 — Assessment: {ASSESSMENT_QUESTIONS} questions (numbers 1-3)", True, (255,220,100))
    else:
        lc = LEVEL_COLORS.get(current_level, (200,200,200))
        banner = font.render(f"Session {session_number} — Level: {LEVEL_LABELS.get(current_level,'')}", True, lc)
    screen.blit(banner, ((WIDTH-banner.get_width())//2, 170))

    for btn in mode_buttons:
        hovered = bool(finger_pos and btn["rect"].collidepoint(finger_pos))
        col     = btn["hover_color"] if hovered else btn["color"]
        pygame.draw.rect(screen, col,           btn["rect"], border_radius=20)
        pygame.draw.rect(screen, (255,255,255), btn["rect"], 3, border_radius=20)
        if hovered:
            pygame.draw.rect(screen, (255,255,200), btn["rect"], 4)
        txt = large_font.render(btn["mode"], True, (255,255,255))
        screen.blit(txt, txt.get_rect(center=btn["rect"].center))
        if hovered and current_pinch:
            GAME_MODE = btn["mode"]
            game_sel_active = False

    pi = small_font.render(f"Player: {player_name} | Age: {player_age} | Grade: {student_grade}", True, (200,200,255))
    screen.blit(pi, (20, HEIGHT-40))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False; game_sel_active = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False; game_sel_active = False
            elif event.key == pygame.K_1:
                GAME_MODE = "COUNTING";  game_sel_active = False
            elif event.key == pygame.K_2:
                GAME_MODE = "ADDITION";  game_sel_active = False

if not GAME_MODE or not running:
    cap.release(); pygame.quit(); sys.exit()


# ─────────────────────────────────────────────────────────────
# INITIALISE  GAME
# ─────────────────────────────────────────────────────────────
def get_next_question(mode: str):
    """Return (target_count, problem_text [,num1, num2])."""
    if mode == "COUNTING":
        target = q_manager.next_number()
        if audio_loaded and target in number_pronunciation and number_pronunciation[target]:
            pygame.time.delay(300)
            play_sound(number_pronunciation[target], allow_interrupt=False)
        return target, str(target)
    else:  # ADDITION
        pool  = LEVEL_NUMBERS.get(current_level, [1,2,3])
        a     = random.choice(pool)
        b     = random.choice(pool)
        # cap sum to 10 for playability
        while a + b > 10:
            a = random.choice(pool); b = random.choice(pool)
        return a+b, f"{a} + {b} = ?", a, b

if GAME_MODE == "COUNTING":
    res_q = get_next_question(GAME_MODE)
    target_count, problem_text = res_q[0], res_q[1]
else:
    target_count, problem_text, num1, num2 = get_next_question(GAME_MODE)

dropped        = 0
score          = 0
total_problems = 0
correct_problems = 0
apples         = [spawn_apple()]

# Interaction state
picked_apple       = None
pinch_active       = False
submit_pinched     = False
last_pinch_time    = 0
smoothed_finger_pos = None
SMOOTH_ALPHA       = 0.5

message        = ""
message_time   = 0
message_duration = 1.5
show_result    = False
result_message = ""
result_color   = (255,255,255)
current_sound  = None

problem_start_time = time.time()
reaction_times     = []

session_id    = datetime.now().strftime('%Y%m%d_%H%M%S_') + str(int(time.time()*1000)%10000)
session_start = datetime.now()
game_start    = time.time()

# Assessment-phase progress
assessment_done = False

# Adaptive phase counter
adaptive_answers_count = 0
last_adaptive_level = None

print(f"\n✓ {GAME_MODE} game started  |  {'ASSESSMENT' if is_assessment_session else 'ADAPTIVE'} phase")
print(f"  Level: {current_level}  |  p_know: {bkt.p_know:.3f}\n")


# ─────────────────────────────────────────────────────────────
# HUD  OVERLAY  HELPERS
# ─────────────────────────────────────────────────────────────
def draw_bkt_hud(screen, bkt: BKTEngine, level: str,
                 is_assessment: bool, q_done: int, total_q: int):
    """
    Session 1 (assessment): top-left shows only the assessment progress counter.
    Session 2+  (adaptive): top-left shows mastery bar + level label.
    """
    hud_x, hud_y = 30, 30

    if is_assessment:
        # ── ASSESSMENT HUD: progress counter only ─────────────
        questions_left = total_q - q_done
        header   = small_font.render("Assessment Phase", True, (255, 220, 100))
        progress = font.render(f"Question  {q_done} / {total_q}", True, (255, 255, 255))
        remain   = tiny_font.render(f"{questions_left} question{'s' if questions_left != 1 else ''} remaining",
                                    True, (200, 200, 200))

        # Background panel sized to content
        panel_w = max(header.get_width(), progress.get_width(), remain.get_width()) + 30
        panel_h = header.get_height() + progress.get_height() + remain.get_height() + 24
        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 170))
        pygame.draw.rect(bg, (255, 220, 100), (0, 0, panel_w, panel_h), 2)
        screen.blit(bg, (hud_x - 10, hud_y - 10))

        screen.blit(header,   (hud_x, hud_y))
        screen.blit(progress, (hud_x, hud_y + header.get_height() + 6))
        screen.blit(remain,   (hud_x, hud_y + header.get_height() + progress.get_height() + 14))

    else:
        # ── ADAPTIVE HUD: enhanced mastery probability panel ────────────
        col         = LEVEL_COLORS.get(level, (200, 200, 200))
        
        # Panel dimensions
        panel_w = 380
        panel_h = 280
        pad = 15
        
        # Background with border
        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        pygame.draw.rect(bg, col, (0, 0, panel_w, panel_h), 3)
        screen.blit(bg, (hud_x - 10, hud_y - 10))
        
        y_offset = hud_y
        
        # Title
        header_surf = small_font.render("MASTERY PROBABILITY", True, (200, 200, 200))
        screen.blit(header_surf, (hud_x, y_offset))
        y_offset += header_surf.get_height() + 8
        
        # Large percentage display
        prob_pct = bkt.p_know * 100
        prob_surf = large_font.render(f"{prob_pct:.0f}%", True, col)
        screen.blit(prob_surf, (hud_x + 10, y_offset))
        y_offset += prob_surf.get_height() + 12
        
        # Progress bar (0 → 100%)
        bar_x = hud_x + 5
        bar_y = y_offset
        bar_w = panel_w - 20
        bar_h = 22
        
        # Background bar
        pygame.draw.rect(screen, (45, 45, 45), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(screen, col, (bar_x, bar_y, bar_w, bar_h), 2)
        
        # Fill based on p_know
        fill_w = int((bkt.p_know / 1.0) * bar_w)
        if fill_w > 0:
            pygame.draw.rect(screen, col, (bar_x + 2, bar_y + 2, fill_w - 4, bar_h - 4))
        
        # Threshold markers
        low_marker_x = bar_x + int((LOW_THRESHOLD / 1.0) * bar_w)
        high_marker_x = bar_x + int((HIGH_THRESHOLD / 1.0) * bar_w)
        pygame.draw.line(screen, (200, 200, 100), (low_marker_x, bar_y - 4), (low_marker_x, bar_y + bar_h + 4), 2)
        pygame.draw.line(screen, (100, 200, 100), (high_marker_x, bar_y - 4), (high_marker_x, bar_y + bar_h + 4), 2)
        
        y_offset += bar_h + 12
        
        # Threshold info
        threshold_text = f"Goal: {HIGH_THRESHOLD*100:.0f}%  |  Current Level: {LEVEL_LABELS.get(level, level)}"
        threshold_surf = tiny_font.render(threshold_text, True, (180, 180, 180))
        screen.blit(threshold_surf, (hud_x + 5, y_offset))
        y_offset += threshold_surf.get_height() + 8
        
        # Status text
        if prob_pct >= HIGH_THRESHOLD * 100:
            status = "✓ ADVANCED - Excellent Progress!"
            status_col = (100, 220, 100)
        elif prob_pct >= LOW_THRESHOLD * 100:
            status = "→ INTERMEDIATE - Keep practicing!"
            status_col = (255, 200, 60)
        else:
            status = "↑ BASIC - Build foundations"
            status_col = (255, 100, 100)
        
        status_surf = tiny_font.render(status, True, status_col)
        screen.blit(status_surf, (hud_x + 5, y_offset))
        y_offset += status_surf.get_height() + 8
        
        # BKT Features mini-display
        feats = bkt.get_features()
        feat_labels = ["Accuracy", "Speed", "Consistency", "Trend"]
        feat_indices = [0, 1, 3, 4]  # Skip p_know (index 2)
        
        feature_title = tiny_font.render("Performance:", True, (160, 160, 160))
        screen.blit(feature_title, (hud_x + 5, y_offset))
        y_offset += feature_title.get_height() + 4
        
        for label, idx in zip(feat_labels, feat_indices):
            val = feats[idx]
            val_col = (100, 220, 100) if val > 0.65 else (255, 200, 60) if val > 0.4 else (220, 100, 100)
            feat_str = f"  {label}: {val*100:.0f}%"
            feat_surf = tiny_font.render(feat_str, True, val_col)
            screen.blit(feat_surf, (hud_x + 5, y_offset))
            y_offset += feat_surf.get_height() + 3
        
        # Adaptive re-eval progress indicator
        y_offset += 4
        if not is_assessment:
            progress_in_cycle = adaptive_answers_count % ADAPTIVE_REEVAL_EVERY
            cycle_num = (adaptive_answers_count // ADAPTIVE_REEVAL_EVERY) + 1
            eval_txt = tiny_font.render(f"Evaluation Cycle #{cycle_num}: {progress_in_cycle}/{ADAPTIVE_REEVAL_EVERY}", 
                                       True, (200, 150, 100))
            screen.blit(eval_txt, (hud_x + 5, y_offset))

def draw_level_badge(screen, level: str, is_assessment: bool):
    """
    Top-right corner badge.
    Session 1: hidden (level not yet determined).
    Session 2+: shows colour-coded level name.
    """
    if is_assessment:
        return   # No level badge during assessment — level not yet known

    col        = LEVEL_COLORS.get(level, (200, 200, 200))
    badge_surf = font.render(LEVEL_LABELS.get(level, level), True, col)
    bx = WIDTH - badge_surf.get_width() - 30
    bg = pygame.Surface((badge_surf.get_width() + 20, badge_surf.get_height() + 10),
                        pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    screen.blit(bg, (bx - 10, 25))
    screen.blit(badge_surf, (bx, 30))


# ─────────────────────────────────────────────────────────────
# MAIN  GAME  LOOP
# ─────────────────────────────────────────────────────────────
while running:
    dt = clock.tick(30) / 1000.0
    TOTAL_SCREEN_TIME = time.time() - game_start

    ok, img = cap.read()
    if not ok: img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    display_camera_fullscreen(screen, img)

    # ── Hand tracking ────────────────────────────────────────
    current_is_pinch = False
    index_pos        = None
    lms = None
    if hand_landmarker is not None:
        try:
            fr = cv2.flip(cv2.resize(img, (WIDTH, HEIGHT)), 1)
            lms = detect_hand_landmarks(fr)
        except Exception:
            lms = None

    if lms:
        rx  = lms.landmark[8].x * WIDTH
        ry  = lms.landmark[8].y * HEIGHT
        if smoothed_finger_pos is None:
            smoothed_finger_pos = (rx, ry)
        else:
            smoothed_finger_pos = (
                SMOOTH_ALPHA*rx + (1-SMOOTH_ALPHA)*smoothed_finger_pos[0],
                SMOOTH_ALPHA*ry + (1-SMOOTH_ALPHA)*smoothed_finger_pos[1],
            )
        index_pos        = (int(smoothed_finger_pos[0]), int(smoothed_finger_pos[1]))
        current_is_pinch = compute_pinch_state(lms, WIDTH, HEIGHT)
        draw_hand_skeleton(screen, lms, WIDTH, HEIGHT)
    else:
        finger_pos = pygame.mouse.get_pos()
        index_pos = finger_pos
        current_is_pinch = bool(pygame.mouse.get_pressed(3)[0])

    # ── Message timeout ──────────────────────────────────────
    if message and time.time()-message_time > message_duration:
        message = ""

    now = time.time()
    hand_over_submit = bool(index_pos and inside(index_pos[0], index_pos[1], submit))

    # ── Pinch START ──────────────────────────────────────────
    if current_is_pinch and not pinch_active:
        if now - last_pinch_time > PINCH_DELAY:
            pinch_active    = True
            last_pinch_time = now

            if show_result:
                # ── Advance to next question ─────────────────
                show_result = False
                apples, dropped = reset_game(apples, dropped)

                # Check assessment completion
                if is_assessment_session and q_manager.assessment_complete():
                    if not assessment_done:
                        assessment_done = True
                        # Classify with BKT model
                        current_level = bkt.classify_level()
                        q_manager.update_level(current_level)
                        print(f"\n🎓 Assessment complete! Level → {current_level}  p_know={bkt.p_know:.3f}")
                        # Show level announcement (reuse result screen briefly)
                        show_result    = True
                        result_message = f"Level: {LEVEL_LABELS.get(current_level, current_level)}"
                        result_color   = LEVEL_COLORS.get(current_level, (255,255,255))
                else:
                    # Get next question
                    if GAME_MODE == "COUNTING":
                        res_q = get_next_question(GAME_MODE)
                        target_count, problem_text = res_q[0], res_q[1]
                    else:
                        target_count, problem_text, num1, num2 = get_next_question(GAME_MODE)
                    problem_start_time = time.time()
                    if current_sound:
                        current_sound.stop(); current_sound = None

            elif hand_over_submit:
                submit_pinched = True
            elif index_pos:
                for apple in apples:
                    if not apple["picked"] and not apple["in_basket"]:
                        cx = apple["x"] + APPLE_SIZE//2
                        cy = apple["y"] + APPLE_SIZE//2
                        if dist((cx, cy), index_pos) < APPLE_SIZE//2:
                            picked_apple = apple
                            break

    # ── Drag apple ───────────────────────────────────────────
    if pinch_active and picked_apple and index_pos:
        picked_apple["x"] = index_pos[0] - APPLE_SIZE//2
        picked_apple["y"] = index_pos[1] - APPLE_SIZE//2

    # ── Pinch RELEASE ────────────────────────────────────────
    if not current_is_pinch and pinch_active:
        pinch_active = False

        if picked_apple:
            cx = picked_apple["x"] + APPLE_SIZE//2
            cy = picked_apple["y"] + APPLE_SIZE//2
            if inside(cx, cy, basket):
                picked_apple["picked"]    = True
                picked_apple["in_basket"] = True
                dropped += 1
                if audio_loaded and dropped in counting_sounds and counting_sounds[dropped]:
                    play_sound(counting_sounds[dropped])
                apples.append(spawn_apple())
            picked_apple = None

        if submit_pinched:
            total_problems += 1
            q_manager.record_answer()
            reaction_time = time.time() - problem_start_time
            reaction_times.append(reaction_time)

            correct = (dropped == target_count)

            # ── BKT UPDATE ────────────────────────────────────
            bkt.update(correct, reaction_time)

            if correct:
                result_message  = "Congratulations!"
                result_color    = (0, 255, 0)
                score          += 10
                correct_problems += 1
                play_sound(congrats_sound)
                current_sound = congrats_sound
            else:
                result_message = f"Try Again!  {dropped} ≠ {target_count}"
                result_color   = (255, 80, 80)
                play_sound(tryagain_sound)
                current_sound = tryagain_sound

            # ── ADAPTIVE RE-EVALUATION (Session 2+) ─────────────
            if not is_assessment_session:
                adaptive_answers_count += 1
                if adaptive_answers_count % ADAPTIVE_REEVAL_EVERY == 0:
                    old_level = current_level
                    new_level = bkt.classify_level()
                    if new_level != old_level:
                        current_level = new_level
                        q_manager.update_level(current_level)
                        # Queue the level change announcement for next result screen
                        result_message = f"Level Up! → {LEVEL_LABELS.get(new_level, new_level)}"
                        result_color = LEVEL_COLORS.get(new_level, (255, 255, 255))
                        print(f"\n  ⬆  ADAPTIVE LEVEL CHANGED: {old_level} → {new_level}")
                        print(f"     p_know={bkt.p_know:.3f} (Answer #{adaptive_answers_count})")
                    else:
                        print(f"  ✓ Adaptive check #{adaptive_answers_count//ADAPTIVE_REEVAL_EVERY}: Level stable at {current_level} (p_know={bkt.p_know:.3f})")
                        # Keep the original result message
                        if correct:
                            result_message = "Congratulations!"
                            result_color = (0, 255, 0)
                        else:
                            result_message = f"Try Again!  {dropped} ≠ {target_count}"
                            result_color = (255, 80, 80)

            log_interaction(
                player_name, player_age, student_grade, GAME_MODE,
                problem_text, dropped, target_count,
                reaction_time, correct, score, session_id,
                TOTAL_SCREEN_TIME, bkt.p_know, current_level,
            )

            show_result    = True
            submit_pinched = False

    # ──────────────────────────────────────────────────────────
    # DRAW  GAME  ELEMENTS
    # ──────────────────────────────────────────────────────────
    # Apples not yet in basket
    for apple in apples:
        if not apple["in_basket"]:
            screen.blit(apple_surface, (apple["x"], apple["y"]))

    # Basket
    bi = min(dropped, len(basket_images)-1)
    screen.blit(basket_images[bi], (basket["x"], basket["y"]))

    # Submit button
    s_col = (0, 180, 0)
    pygame.draw.rect(screen, (0,140,0),  (submit["x"]+5, submit["y"]+5, submit["w"], submit["h"]))
    pygame.draw.rect(screen, s_col,      (submit["x"],   submit["y"],   submit["w"], submit["h"]))
    pygame.draw.rect(screen, (0,220,0),  (submit["x"],   submit["y"],   submit["w"], 10))
    if hand_over_submit:
        pygame.draw.rect(screen, (255,255,0), (submit["x"], submit["y"], submit["w"], submit["h"]), 3)
    screen.blit(font.render("SUBMIT", True, (255,255,255)), (submit["x"]+40, submit["y"]+30))

    # Problem display
    if GAME_MODE == "COUNTING":
        nt = number_font.render(str(target_count), True, (255,255,200))
        screen.blit(nt, nt.get_rect(center=(WIDTH//2, 100)))
    else:
        pt = large_font.render(problem_text, True, (255,200,150))
        screen.blit(pt, pt.get_rect(center=(WIDTH//2, 100)))

    # Mini apple progress row
    apple_y   = 210
    start_ax  = (WIDTH - target_count*90)//2
    for i in range(target_count):
        ax   = start_ax + i*90
        rect = small_apple_surface.get_rect(center=(ax, apple_y))
        if i < dropped:
            screen.blit(small_apple_surface, rect)
            if GAME_MODE == "COUNTING":
                nt2 = small_font.render(str(i+1), True, (255,255,255))
                screen.blit(nt2, nt2.get_rect(center=(ax, apple_y)))
        else:
            screen.blit(empty_apple_surface, rect)

    # Score
    sg = pygame.Surface((260, 60), pygame.SRCALPHA)
    sg.fill((0,0,0,150))
    screen.blit(sg, (WIDTH-290, 120))
    screen.blit(font.render(f"Score: {score}", True, (255,255,0)), (WIDTH-280, 130))

    # BKT HUD
    draw_bkt_hud(screen, bkt, current_level,
                 is_assessment_session, q_manager.questions_done, ASSESSMENT_QUESTIONS)
    draw_level_badge(screen, current_level, is_assessment_session)

    # Player info & instructions
    info = tiny_font.render(
        f"Player: {player_name[:12]} | Session #{session_number} | Time: {TOTAL_SCREEN_TIME:.0f}s",
        True, (200,200,255))
    screen.blit(info, (WIDTH-info.get_width()-20, HEIGHT-40))

    instr_txt = ("ASSESSMENT phase – answer all 15 questions!" if is_assessment_session and not assessment_done
                 else "Pinch apple → drag to basket → SUBMIT")
    screen.blit(tiny_font.render(instr_txt, True, (255,255,255)), (30, HEIGHT-50))
    screen.blit(tiny_font.render(f"Audio: {'ON' if audio_loaded else 'OFF'}", True,
                                  (100,255,100) if audio_loaded else (255,100,100)), (30, HEIGHT-80))

    # Result overlay
    if show_result:
        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ov.fill((0,0,0,180))
        screen.blit(ov, (0,0))

        rs = large_font.render(result_message, True, result_color)
        screen.blit(rs, rs.get_rect(center=(WIDTH//2, HEIGHT//2-80)))

        if is_assessment_session and not assessment_done:
            # Assessment in progress: show question counter, NOT level/mastery
            q_done = q_manager.questions_done
            prog = font.render(f"Question {q_done} of {ASSESSMENT_QUESTIONS}", True, (255, 220, 100))
            screen.blit(prog, prog.get_rect(center=(WIDTH//2, HEIGHT//2)))
        else:
            # Adaptive (or assessment-complete announcement): show mastery + level
            pk_txt = font.render(
                f"Mastery: {bkt.p_know:.0%}  |  {LEVEL_LABELS.get(current_level, '')}",
                True, (255, 255, 200))
            screen.blit(pk_txt, pk_txt.get_rect(center=(WIDTH//2, HEIGHT//2)))

        cs = font.render("Pinch anywhere to continue", True, (255,255,255))
        screen.blit(cs, cs.get_rect(center=(WIDTH//2, HEIGHT//2+90)))

    pygame.display.update()

    # ── Events ────────────────────────────────────────────────
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif show_result and event.key == pygame.K_SPACE:
                show_result = False
                apples, dropped = reset_game(apples, dropped)
                if GAME_MODE == "COUNTING":
                    res_q = get_next_question(GAME_MODE)
                    target_count, problem_text = res_q[0], res_q[1]
                else:
                    target_count, problem_text, num1, num2 = get_next_question(GAME_MODE)
                problem_start_time = time.time()
                if current_sound:
                    current_sound.stop(); current_sound = None
            elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3,
                                pygame.K_4, pygame.K_5, pygame.K_6,
                                pygame.K_7, pygame.K_8, pygame.K_9]:
                n = event.key - pygame.K_0
                if audio_loaded and n in counting_sounds and counting_sounds[n]:
                    play_sound(counting_sounds[n])


# ─────────────────────────────────────────────────────────────
# SESSION  END  – SAVE  &  LOG
# ─────────────────────────────────────────────────────────────
# Final level classification
final_level  = bkt.classify_level()
final_p_know = bkt.p_know

avg_rt = sum(reaction_times)/len(reaction_times) if reaction_times else None

log_session_end(
    player_name, player_age, student_grade, GAME_MODE,
    session_start, score, total_problems, correct_problems,
    session_id, TOTAL_SCREEN_TIME, avg_rt,
    final_p_know, final_level, session_number,
)

save_student_data(player_key, final_level, final_p_know, session_number)

print("\n" + "="*55)
print("SESSION SUMMARY")
print("="*55)
print(f"  Player:          {player_name}")
print(f"  Session #:       {session_number}  ({'ASSESSMENT' if is_assessment_session else 'ADAPTIVE'})")
print(f"  Game Mode:       {GAME_MODE}")
print(f"  Final Score:     {score}")
print(f"  Problems:        {total_problems}  ({correct_problems} correct)")
if total_problems > 0:
    print(f"  Accuracy:        {correct_problems/total_problems*100:.1f}%")
print(f"  Duration:        {TOTAL_SCREEN_TIME:.1f}s")
print(f"  Final p_know:    {final_p_know:.3f}")
print(f"  Final Level:     {final_level}")
if avg_rt:
    print(f"  Avg React. Time: {avg_rt:.2f}s")
print("="*55)

cap.release()
pygame.quit()
sys.exit(0)
