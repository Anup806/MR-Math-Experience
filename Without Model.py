import cv2
import mediapipe as mp
import random
import time
import numpy as np
import threading
import math
import sys
import csv
import os
from datetime import datetime
from urllib.request import urlretrieve
from types import SimpleNamespace

try:
    import pygame
except Exception:
    print("Error: pygame is not installed in this Python environment.")
    print("Install it with:")
    print("    python -m pip install pygame")
    print("If you use a virtualenv/conda, activate it first.")
    sys.exit(1)

# Try to import TensorFlow; if not available, disable DKT gracefully
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

# -----------------------------
# INITIAL SETUP
pygame.init()

# Initialize pygame mixer for sound with better settings
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)  # Increased buffer
    print("Audio system initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize audio system: {e}")
    print("Game will continue without sound effects")

# Get desktop size
try:
    WIDTH, HEIGHT = pygame.display.get_desktop_sizes()[0]
except Exception:
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h

# Set up fullscreen display
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
pygame.display.set_caption("Math Learning – Drag Apples to Basket Game")
pygame.event.set_blocked(pygame.MOUSEMOTION)
pygame.mouse.set_visible(True)

# Font setup
font = pygame.font.Font(None, 60)
large_font = pygame.font.Font(None, 100)
small_font = pygame.font.Font(None, 36)
tiny_font = pygame.font.Font(None, 24)  # Added for smaller instructions
number_font = pygame.font.Font(None, 300)  # Very large font for counting numbers
clock = pygame.time.Clock()

# MediaPipe task-based hand tracking setup
load_hand_landmarker()

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Check camera connection.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# -----------------------------
# CSV LOGGING SETUP
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(SCRIPT_DIR, "photos")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
CSV_DIR = os.path.join(SCRIPT_DIR, "game_data")
INTERACTIONS_CSV = os.path.join(CSV_DIR, "interactions.csv")
SESSIONS_CSV = os.path.join(CSV_DIR, "sessions.csv")

# CSV Headers
INTERACTIONS_HEADERS = [
    'student_name', 
    'age', 
    'student_grade',
    'game_mode',
    'timestamp', 
    'math_problem', 
    'user_answer', 
    'correct_answer',
    'reaction_time_s', 
    'correct', 
    'score', 
    'session_id',
    'total_screen_time'
]

SESSIONS_HEADERS = [
    'student_name', 
    'age', 
    'student_grade',
    'game_mode',
    'session_start', 
    'session_end', 
    'final_score', 
    'total_problems', 
    'correct_problems', 
    'accuracy', 
    'session_id',
    'total_screen_time',
    'average_reaction_time'
]

def initialize_csv_files():
    """Create CSV files with headers if they don't exist."""
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(CSV_DIR):
            os.makedirs(CSV_DIR)
        
        # Initialize interactions CSV
        if not os.path.exists(INTERACTIONS_CSV):
            with open(INTERACTIONS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(INTERACTIONS_HEADERS)
        
        # Initialize sessions CSV
        if not os.path.exists(SESSIONS_CSV):
            with open(SESSIONS_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(SESSIONS_HEADERS)
        
        print(f"CSV files initialized in: {CSV_DIR}")
        
    except Exception as e:
        print(f"Error initializing CSV files: {e}")

# Initialize CSV files
initialize_csv_files()

def log_interaction(player_name, age, student_grade, game_mode, math_problem, user_answer, correct_answer,
                    reaction_time_s, correct, current_score, session_id, total_screen_time):
    """Log each math problem interaction to CSV."""
    try:
        with open(INTERACTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                player_name,
                age,
                student_grade,
                game_mode,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                math_problem,
                user_answer,
                correct_answer,
                f"{reaction_time_s:.3f}" if reaction_time_s is not None else "",
                1 if correct else 0,
                current_score,
                session_id,
                f"{total_screen_time:.2f}"
            ])
    except Exception as e:
        print(f"Error logging interaction: {e}")

def log_session_end(player_name, age, student_grade, game_mode, session_start, final_score, 
                    total_problems, correct_problems, session_id,
                    total_screen_time, average_reaction_time):
    """Log session summary to CSV."""
    try:
        accuracy = (correct_problems / total_problems * 100) if total_problems > 0 else 0
        with open(SESSIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                player_name,
                age,
                student_grade,
                game_mode,
                session_start.strftime('%Y-%m-%d %H:%M:%S'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                final_score,
                total_problems,
                correct_problems,
                f"{accuracy:.2f}",
                session_id,
                f"{total_screen_time:.2f}",
                f"{average_reaction_time:.3f}" if average_reaction_time is not None else ""
            ])
    except Exception as e:
        print(f"Error logging session: {e}")

# -----------------------------
# GAME VARIABLES
# -----------------------------
# Apple and game variables
APPLE_SIZE = 300
SPAWN_X, SPAWN_Y = 120, 300
basket = {"x": (WIDTH - 600) // 2, "y": HEIGHT - 460, "w": 600, "h": 400}
submit = {"x": WIDTH - 380, "y": HEIGHT - 380, "w": 220, "h": 90}

# Game modes
GAME_MODE = None  # Will be set to "COUNTING" or "ADDITION"
GAME_MODES = ["COUNTING", "ADDITION"]

# Pinch parameters
PINCH_THRESHOLD = 45
RELEASE_THRESHOLD = 65
PINCH_DELAY = 0.3

# Message display
message = ""
message_time = 0
message_duration = 1.5

# Result screen
show_result = False
result_message = ""
result_color = (255, 255, 255)
current_sound = None

# Load apple image
try:
    apple_surface = pygame.image.load(os.path.join(PHOTO_DIR, "apple.png")).convert_alpha()
    apple_surface = pygame.transform.scale(apple_surface, (APPLE_SIZE, APPLE_SIZE))
    small_apple_surface = pygame.transform.scale(apple_surface, (80, 80))
    # Create semi-transparent version for empty slots
    empty_apple_surface = small_apple_surface.copy()
    empty_apple_surface.fill((255, 255, 255, 100), special_flags=pygame.BLEND_RGBA_MULT)
except:
    # Create a simple red apple if image is not found
    apple_surface = pygame.Surface((APPLE_SIZE, APPLE_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(apple_surface, (255, 50, 50), (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 10)
    pygame.draw.circle(apple_surface, (200, 30, 30), (APPLE_SIZE//2, APPLE_SIZE//2), APPLE_SIZE//2 - 10, 3)
    stem_rect = pygame.Rect(APPLE_SIZE//2 - 5, APPLE_SIZE//4 - 10, 10, APPLE_SIZE//4)
    pygame.draw.rect(apple_surface, (100, 70, 20), stem_rect)
    leaf_rect = pygame.Rect(APPLE_SIZE//2 + 5, APPLE_SIZE//4 - 15, 15, 10)
    pygame.draw.ellipse(apple_surface, (100, 200, 50), leaf_rect)
    small_apple_surface = pygame.transform.scale(apple_surface, (80, 80))
    empty_apple_surface = small_apple_surface.copy()
    empty_apple_surface.fill((255, 255, 255, 100), special_flags=pygame.BLEND_RGBA_MULT)

# Load basket images
basket_images = []
for i in range(10):
    try:
        basket_img = pygame.image.load(os.path.join(PHOTO_DIR, f"basket{i}.png")).convert_alpha()
        basket_img = pygame.transform.scale(basket_img, (basket["w"], basket["h"]))
        basket_images.append(basket_img)
    except:
        # Create simple basket if images not found
        surf = pygame.Surface((basket["w"], basket["h"]), pygame.SRCALPHA)
        # Basket body
        pygame.draw.ellipse(surf, (210, 180, 140), (10, basket["h"]//2, basket["w"]-20, basket["h"]//2))
        # Basket handle
        pygame.draw.arc(surf, (160, 120, 80), (basket["w"]//4, 10, basket["w"]//2, 50), 
                       math.pi, 2*math.pi, 5)
        # Apples in basket
        if i > 0:
            for j in range(i):
                x_pos = basket["w"]//3 + (j % 3) * 40
                y_pos = basket["h"]//2 + 30 + (j // 3) * 40
                pygame.draw.circle(surf, (255, 50, 50), (x_pos, y_pos), 15)
                pygame.draw.circle(surf, (200, 30, 30), (x_pos, y_pos), 15, 2)
        basket_images.append(surf)

# -----------------------------
# ENHANCED AUDIO SYSTEM WITH COUNTING SOUNDS
# -----------------------------
print("\n" + "="*50)
print("LOADING AUDIO FILES")
print("="*50)

# Load audio files with counting sounds
audio_loaded = False
congrats_sound = None
tryagain_sound = None
counting_sounds = {}  # Dictionary to store counting sounds 1-9
number_pronunciation = {}  # For pronouncing target numbers in COUNTING mode

try:
    print("\nLoading game audio files...")
    
    # Try multiple extensions for each sound file
    def try_load_sound(filename_base, extensions=['.MP3', '.WAV', '.OGG', '.mp3', '.wav']):
        for ext in extensions:
            try:
                filename = os.path.join(AUDIO_DIR, filename_base + ext)
                sound = pygame.mixer.Sound(filename)
                print(f"  ✓ Loaded: {filename}")
                return sound
            except:
                continue
        print(f"  ✗ Failed to load: {filename_base}[{', '.join(extensions)}]")
        return None
    
    # Load congratulation and try again sounds
    congrats_sound = try_load_sound("Cheer")
    if not congrats_sound:
        congrats_sound = try_load_sound("congratulation")
    tryagain_sound = try_load_sound("tryagain")
    
    # Load counting sounds (1.MP3 to 9.MP3)
    print("\nLoading counting sounds (1-9):")
    for i in range(1, 10):
        counting_sounds[i] = try_load_sound(str(i))
        if counting_sounds[i]:
            counting_sounds[i].set_volume(0.6)  # Set volume for counting sounds
    
    # Load number pronunciation sounds (optional)
    print("\nLoading number pronunciation sounds:")
    for i in range(1, 10):
        number_pronunciation[i] = try_load_sound(f"number_{i}")
        if number_pronunciation[i]:
            number_pronunciation[i].set_volume(0.7)
        else:
            # Fall back to counting sounds if specific files don't exist
            number_pronunciation[i] = counting_sounds[i]
    
    # Check if at least some sounds were loaded
    sounds_loaded = sum(1 for s in counting_sounds.values() if s is not None)
    if sounds_loaded > 0 or congrats_sound or tryagain_sound:
        audio_loaded = True
        print(f"\n✓ Audio system ready. Loaded {sounds_loaded} counting sounds.")
        
        # Create backup synthesized sounds if files are missing
        if sounds_loaded < 9:
            print("  Note: Some counting sounds are missing. Game will use available sounds.")
    else:
        audio_loaded = False
        print("\n✗ No audio files could be loaded. Game will run without sound.")
    
    print("="*50 + "\n")
    
except Exception as e:
    print(f"\n✗ Error in audio initialization: {e}")
    import traceback
    traceback.print_exc()
    audio_loaded = False

# Enhanced sound playing function
def play_sound(sound, allow_interrupt=True, volume=None):
    """Play a sound with optional interruption control."""
    if not audio_loaded or sound is None:
        return False
    
    try:
        # Save original volume
        original_volume = sound.get_volume()
        
        # Apply custom volume if specified
        if volume is not None:
            sound.set_volume(volume)
        
        # Play the sound
        if allow_interrupt:
            sound.stop()  # Stop any current playback of this sound
            sound.play()
        else:
            # Only play if not already playing
            if not pygame.mixer.get_busy():
                sound.play()
        
        # Restore original volume
        if volume is not None:
            sound.set_volume(original_volume)
        
        return True
    except Exception as e:
        print(f"Error playing sound: {e}")
        return False

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def dist(a, b):
    """Calculate Euclidean distance between two points"""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def inside(x, y, rect):
    """Check if point (x, y) is inside rectangle"""
    return rect["x"] < x < rect["x"] + rect["w"] and rect["y"] < y < rect["y"] + rect["h"]

def compute_pinch_state(hand_landmarks, img_w, img_h):
    """Check if thumb and index finger are pinching"""
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    
    tx, ty = thumb.x * img_w, thumb.y * img_h
    ix, iy = index.x * img_w, index.y * img_h
    
    distance = math.hypot(tx - ix, ty - iy)
    return distance < PINCH_THRESHOLD

def get_finger_positions(hand_landmarks, img_w, img_h):
    """Get thumb and index finger positions"""
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    
    thumb_pos = (int(thumb.x * img_w), int(thumb.y * img_h))
    index_pos = (int(index.x * img_w), int(index.y * img_h))
    
    return thumb_pos, index_pos

def draw_hand_skeleton(screen, hand_landmarks, display_w, display_h):
    """Draw hand landmarks and connections"""
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    
    for start, end in connections:
        s = hand_landmarks.landmark[start]
        e = hand_landmarks.landmark[end]
        sx, sy = int(s.x * display_w), int(s.y * display_h)
        ex, ey = int(e.x * display_w), int(e.y * display_h)
        pygame.draw.line(screen, (0, 255, 0), (sx, sy), (ex, ey), 2)
    
    # Draw landmarks
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * display_w), int(lm.y * display_h)
        pygame.draw.circle(screen, (255, 0, 0), (x, y), 4)
    
    # Draw pinch line between thumb and index
    thumb_pos, index_pos = get_finger_positions(hand_landmarks, display_w, display_h)
    pygame.draw.line(screen, (255, 255, 0), thumb_pos, index_pos, 2)

# -----------------------------
# CAMERA DISPLAY FUNCTIONS
# -----------------------------
def cvimage_to_pygame(image):
    """Convert cv2 image to PyGame surface."""
    if image is None:
        return None
    try:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        return pygame.image.frombuffer(image.tobytes(), image.shape[1::-1], "RGB")
    except Exception as e:
        print(f"Error converting image to pygame: {e}")
        return None

def display_camera_fullscreen(screen, img):
    """Display camera feed as full-screen background."""
    if img is not None:
        try:
            camera_frame = cv2.resize(img, (WIDTH, HEIGHT))
            camera_surface = cvimage_to_pygame(camera_frame)
            
            if camera_surface is not None:
                screen.blit(camera_surface, (0, 0))
                
                # Semi-transparent overlay
                overlay_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay_surf.fill((0, 0, 0, 40))
                screen.blit(overlay_surf, (0, 0))
                
        except Exception as e:
            print(f"Error displaying camera: {e}")
            screen.fill((0, 0, 0))
    else:
        screen.fill((0, 0, 0))

# -----------------------------
# GAME MODE FUNCTIONS
# -----------------------------
counting_question_queue = []

def new_counting_question():
    """Generate a new counting question with 1-5 range, 50 questions total (10 of each)."""
    global counting_question_queue
    
    # If queue is empty, populate it with 50 questions (10 of each, 1-5)
    if not counting_question_queue:
        # Create list with 10 copies of each number from 1 to 5
        counting_question_queue = []
        for i in range(1, 6):
            counting_question_queue.extend([i] * 10)
        
        random.shuffle(counting_question_queue)
        print("Generated new batch of 50 counting questions (10 each of 1-5)")
    
    # Pop the next question
    target = counting_question_queue.pop(0)
    
    # Play the target number pronunciation if available (only for COUNTING mode)
    if audio_loaded and target in number_pronunciation and number_pronunciation[target] is not None:
        # Play after a short delay
        pygame.time.delay(300)  # 300ms delay
        play_sound(number_pronunciation[target], allow_interrupt=False)
        print(f"Playing target number: {target}")
    
    return target, f"{target}"  # Just return the number, no "Count to"

def new_addition_question():
    """Generate a new addition question"""
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    target = a + b
    
    return target, f"{a} + {b} = ?", a, b

def spawn_new_apple():
    """Create a new apple at spawn position"""
    return {
        "x": SPAWN_X,
        "y": SPAWN_Y,
        "picked": False,
        "in_basket": False,
        "has_been_counted": False  # Track if we've played counting sound for this apple
    }

def reset_game():
    """Reset game for new problem"""
    global apples, message, message_time, show_result, current_sound, dropped
    dropped = 0
    apples = [spawn_new_apple()]
    message = ""
    message_time = 0
    show_result = False
    if current_sound:
        current_sound.stop()
    current_sound = None

# -----------------------------
# PLAYER INPUT SCREEN
# -----------------------------
player_name = ""
player_age = ""
student_grade = ""
name_entry_active = True
running = False
input_focus = "name"

while name_entry_active:
    dt = clock.tick(30) / 1000.0
    
    screen.fill((30, 30, 40))
    
    # Title
    title_txt = large_font.render("Math Learning Game", True, (100, 200, 255))
    screen.blit(title_txt, ((WIDTH - title_txt.get_width())//2, HEIGHT//6))
    
    # Instruction
    instr_txt = font.render("Enter your details:", True, (255, 255, 255))
    screen.blit(instr_txt, ((WIDTH - instr_txt.get_width())//2, HEIGHT//4))
    
    # Calculate positions
    box_w, box_h = 500, 60
    box_x = (WIDTH - box_w) // 2
    start_y = HEIGHT//2 - 120
    
    # Name input
    name_label = small_font.render("Name:", True, (200, 200, 200))
    screen.blit(name_label, (box_x, start_y))
    
    pygame.draw.rect(screen, (50, 50, 60), (box_x, start_y + 30, box_w, box_h))
    name_border_color = (100, 150, 255) if input_focus == "name" else (80, 120, 200)
    pygame.draw.rect(screen, name_border_color, (box_x, start_y + 30, box_w, box_h), 3)
    
    name_txt = font.render(player_name if player_name else "_", True, (255, 255, 255))
    screen.blit(name_txt, (box_x + 20, start_y + 40))
    
    # Age input
    age_label = small_font.render("Age:", True, (200, 200, 200))
    screen.blit(age_label, (box_x, start_y + 110))
    
    pygame.draw.rect(screen, (50, 50, 60), (box_x, start_y + 140, box_w, box_h))
    age_border_color = (100, 150, 255) if input_focus == "age" else (80, 120, 200)
    pygame.draw.rect(screen, age_border_color, (box_x, start_y + 140, box_w, box_h), 3)
    
    age_txt = font.render(player_age if player_age else "_", True, (255, 255, 255))
    screen.blit(age_txt, (box_x + 20, start_y + 150))
    
    # Grade input
    grade_label = small_font.render("Grade/Level:", True, (200, 200, 200))
    screen.blit(grade_label, (box_x, start_y + 220))
    
    pygame.draw.rect(screen, (50, 50, 60), (box_x, start_y + 250, box_w, box_h))
    grade_border_color = (100, 150, 255) if input_focus == "grade" else (80, 120, 200)
    pygame.draw.rect(screen, grade_border_color, (box_x, start_y + 250, box_w, box_h), 3)
    
    grade_txt = font.render(student_grade if student_grade else "_", True, (255, 255, 255))
    screen.blit(grade_txt, (box_x + 20, start_y + 260))
    
    # Instructions
    hint_txt = small_font.render("Press TAB to switch fields | ENTER to continue", True, (150, 200, 150))
    screen.blit(hint_txt, ((WIDTH - hint_txt.get_width())//2, HEIGHT - 100))
    
    pygame.display.update()
    
    # Handle input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            name_entry_active = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if player_name.strip() and player_age.strip() and student_grade.strip():
                    name_entry_active = False
                    running = True
            elif event.key == pygame.K_BACKSPACE:
                if input_focus == "name":
                    player_name = player_name[:-1]
                elif input_focus == "age":
                    player_age = player_age[:-1]
                elif input_focus == "grade":
                    student_grade = student_grade[:-1]
            elif event.key == pygame.K_TAB:
                if input_focus == "name":
                    input_focus = "age"
                elif input_focus == "age":
                    input_focus = "grade"
                else:
                    input_focus = "name"
            elif event.unicode:
                if input_focus == "name" and len(player_name) < 30:
                    if event.unicode.isprintable():
                        player_name += event.unicode
                elif input_focus == "age" and len(player_age) < 3:
                    if event.unicode.isdigit():
                        player_age += event.unicode
                elif input_focus == "grade" and len(student_grade) < 50:
                    if event.unicode.isprintable():
                        student_grade += event.unicode

if not running:
    cap.release()
    pygame.quit()
    sys.exit()

# -----------------------------
# GAME SELECTION SCREEN
# -----------------------------
game_selection_active = True
selected_mode = None

# Game mode buttons
mode_buttons = []
button_width, button_height = 400, 150
button_spacing = 100
total_width = button_width * 2 + button_spacing
start_x = (WIDTH - total_width) // 2
y_pos = HEIGHT // 2 - button_height // 2

for i, mode in enumerate(GAME_MODES):
    btn_x = start_x + i * (button_width + button_spacing)
    mode_buttons.append({
        "mode": mode,
        "rect": pygame.Rect(btn_x, y_pos, button_width, button_height),
        "color": (70, 130, 180) if mode == "COUNTING" else (60, 179, 113),
        "hover_color": (100, 160, 210) if mode == "COUNTING" else (85, 199, 133),
        "text_color": (255, 255, 255)
    })

while game_selection_active and running:
    dt = clock.tick(30) / 1000.0
    
    # Get camera frame
    success, img = cap.read()
    if not success:
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Display camera background
    display_camera_fullscreen(screen, img)
    
    # Process hand detection
    finger_pos = None
    hand_detected = False
    current_is_pinch = False
    
    handLms = None
    if hand_landmarker is not None:
        try:
            frame = cv2.resize(img, (WIDTH, HEIGHT))
            frame = cv2.flip(frame, 1)
            handLms = detect_hand_landmarks(frame)
        except Exception:
            handLms = None

    if handLms:
        hand_detected = True
        
        # Get fingertip position
        index_x = int(handLms.landmark[8].x * WIDTH)
        index_y = int(handLms.landmark[8].y * HEIGHT)
        finger_pos = (index_x, index_y)
        
        # Draw hand skeleton
        draw_hand_skeleton(screen, handLms, WIDTH, HEIGHT)
        
        # Detect pinch
        current_is_pinch = compute_pinch_state(handLms, WIDTH, HEIGHT)
    else:
        finger_pos = pygame.mouse.get_pos()
        current_is_pinch = bool(pygame.mouse.get_pressed(3)[0])
    
    # Draw title
    title_bg = pygame.Surface((WIDTH, 120), pygame.SRCALPHA)
    title_bg.fill((0, 0, 0, 150))
    screen.blit(title_bg, (0, 50))
    
    title_txt = large_font.render("SELECT GAME MODE", True, (255, 255, 200))
    screen.blit(title_txt, ((WIDTH - title_txt.get_width())//2, 80))
    
    # Draw game mode buttons
    for button in mode_buttons:
        # Check if hand is over button
        is_hovered = False
        if finger_pos and button["rect"].collidepoint(finger_pos):
            is_hovered = True
            # Draw hover effect
            pygame.draw.rect(screen, (255, 255, 200), button["rect"], 4)
        
        # Draw button
        button_color = button["hover_color"] if is_hovered else button["color"]
        pygame.draw.rect(screen, button_color, button["rect"], border_radius=20)
        pygame.draw.rect(screen, (255, 255, 255), button["rect"], 3, border_radius=20)
        
        # Draw button text
        text = large_font.render(button["mode"], True, button["text_color"])
        text_rect = text.get_rect(center=button["rect"].center)
        screen.blit(text, text_rect)
        
        # Handle pinch selection
        if is_hovered and current_is_pinch:
            selected_mode = button["mode"]
            GAME_MODE = selected_mode
            game_selection_active = False
    
    # Draw player info
    player_info = small_font.render(f"Player: {player_name} | Age: {player_age} | Grade: {student_grade}", True, (200, 200, 255))
    screen.blit(player_info, (20, HEIGHT - 40))
    
    pygame.display.update()
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            game_selection_active = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                game_selection_active = False
            elif event.key == pygame.K_1:
                selected_mode = "COUNTING"
                GAME_MODE = selected_mode
                game_selection_active = False
            elif event.key == pygame.K_2:
                selected_mode = "ADDITION"
                GAME_MODE = selected_mode
                game_selection_active = False

# If no mode selected or window closed, exit
if not selected_mode or not running:
    cap.release()
    pygame.quit()
    sys.exit()

# -----------------------------
# INITIALIZE SELECTED GAME
# -----------------------------
print(f"\nStarting {GAME_MODE} game...")
print(f"Player: {player_name}, Age: {player_age}, Grade: {student_grade}")

# Initialize game variables based on selected mode
if GAME_MODE == "COUNTING":
    target_count, problem_text = new_counting_question()
elif GAME_MODE == "ADDITION":
    target_count, problem_text, num1, num2 = new_addition_question()

dropped = 0
score = 0
total_problems = 0
correct_problems = 0
apples = [spawn_new_apple()]

# Game state variables
picked_apple = None
pinch_active = False
submit_pinched = False
last_pinch_time = 0
smoothed_finger_pos = None
SMOOTH_ALPHA = 0.5

# Sound tracking
last_counting_sound_time = 0
COUNTING_SOUND_COOLDOWN = 0.5  # Minimum time between counting sounds (seconds)

# Reaction time tracking
problem_start_time = time.time()
reaction_times = []

# Session tracking
session_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + str(int(time.time() * 1000) % 10000)
session_start = datetime.now()
game_start_time = time.time()

# -----------------------------
# MAIN GAME LOOP WITH PINCH-ONLY AUDIO
# -----------------------------
print("\nStarting main game loop...")
print("Controls: Pinch apples and drag to basket. Listen for number when dropped. Pinch SUBMIT when done.")
print("Press ESC to exit, M to return to menu, 1-9 to test counting sounds.\n")

while running:
    dt = clock.tick(30) / 1000.0
    
    # Calculate total screen time
    TOTAL_SCREEN_TIME = time.time() - game_start_time
    
    # Get camera frame
    success, img = cap.read()
    if not success:
        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Display camera background
    display_camera_fullscreen(screen, img)
    
    # Process hand detection
    finger_pos = None
    handLms = None
    if hand_landmarker is not None:
        try:
            frame = cv2.resize(img, (WIDTH, HEIGHT))
            frame = cv2.flip(frame, 1)
            handLms = detect_hand_landmarks(frame)
        except Exception:
            handLms = None
    
    current_is_pinch = False
    index_pos = None
    
    if handLms:
        # Get fingertip position
        raw_x = handLms.landmark[8].x * WIDTH
        raw_y = handLms.landmark[8].y * HEIGHT
        
        if smoothed_finger_pos is None:
            smoothed_finger_pos = (raw_x, raw_y)
        else:
            sx = SMOOTH_ALPHA * raw_x + (1.0 - SMOOTH_ALPHA) * smoothed_finger_pos[0]
            sy = SMOOTH_ALPHA * raw_y + (1.0 - SMOOTH_ALPHA) * smoothed_finger_pos[1]
            smoothed_finger_pos = (sx, sy)
        
        finger_pos = (int(smoothed_finger_pos[0]), int(smoothed_finger_pos[1]))
        index_pos = finger_pos
        
        # Detect pinch
        current_is_pinch = compute_pinch_state(handLms, WIDTH, HEIGHT)
        
        # Draw hand skeleton
        draw_hand_skeleton(screen, handLms, WIDTH, HEIGHT)
    else:
        finger_pos = pygame.mouse.get_pos()
        index_pos = finger_pos
        current_is_pinch = bool(pygame.mouse.get_pressed(3)[0])
    
    # Clear message after duration
    if message and time.time() - message_time > message_duration:
        message = ""
    
    now = time.time()
    
    # Check if hand is over submit button
    hand_over_submit = index_pos and inside(index_pos[0], index_pos[1], submit)
    submit_highlight = hand_over_submit
    
    # Pinch start - AUDIO PLAYS ONLY HERE WHEN PINCHING APPLES
    if current_is_pinch and not pinch_active:
        if now - last_pinch_time > PINCH_DELAY:
            pinch_active = True
            last_pinch_time = now
            
            # If showing result, continue to next problem
            if show_result:
                show_result = False
                reset_game()
                # Generate new question based on game mode
                if GAME_MODE == "COUNTING":
                    target_count, problem_text = new_counting_question()
                else:  # ADDITION
                    target_count, problem_text, num1, num2 = new_addition_question()
                problem_start_time = time.time()
                # Stop sound if playing
                if current_sound:
                    current_sound.stop()
                    current_sound = None
            elif hand_over_submit:
                submit_pinched = True
                # Visual feedback for submit button pinch
                pygame.draw.rect(screen, (0, 255, 0), 
                               (submit["x"], submit["y"], submit["w"], submit["h"]), 3)
            elif index_pos:
                # Check if pinching an apple
                for apple in apples:
                    if not apple["picked"] and not apple["in_basket"]:
                        # Check distance to apple center
                        apple_center_x = apple["x"] + APPLE_SIZE // 2
                        apple_center_y = apple["y"] + APPLE_SIZE // 2
                        if dist((apple_center_x, apple_center_y), index_pos) < APPLE_SIZE // 2:
                            picked_apple = apple
                            
                            # Visual feedback for apple pickup
                            pygame.draw.circle(screen, (0, 255, 0), 
                                             (apple_center_x, apple_center_y), 
                                             APPLE_SIZE // 2, 2)
                            break
    
    # Move apple if pinching one
    if pinch_active and picked_apple and index_pos:
        picked_apple["x"] = index_pos[0] - APPLE_SIZE // 2
        picked_apple["y"] = index_pos[1] - APPLE_SIZE // 2
    
    # Release pinch (check for release) - NO AUDIO HERE
    if not current_is_pinch and pinch_active:
        pinch_active = False
        
        if picked_apple:
            # Check if apple is dropped in basket
            apple_center_x = picked_apple["x"] + APPLE_SIZE // 2
            apple_center_y = picked_apple["y"] + APPLE_SIZE // 2
            
            if inside(apple_center_x, apple_center_y, basket):
                picked_apple["picked"] = True
                picked_apple["in_basket"] = True
                dropped += 1
                
                # PLAY COUNTING SOUND WHEN DROPPED IN BASKET
                if audio_loaded:
                    # Play sound for the current number of apples in basket
                    if dropped in counting_sounds and counting_sounds[dropped] is not None:
                        print(f"Playing counting sound: {dropped}")
                        play_sound(counting_sounds[dropped], allow_interrupt=True)
                
                print(f"Apple dropped in basket. Total: {dropped}/{target_count}")
                
                # Spawn a new apple after successfully dropping one (CONTINUOUS SPAWN)
                new_apple = spawn_new_apple()
                apples.append(new_apple)
                
            picked_apple = None
        
        # Submit if submit button was pinched and released
        if submit_pinched:
            total_problems += 1
            
            # Calculate reaction time
            reaction_time = time.time() - problem_start_time
            reaction_times.append(reaction_time)
            
            correct = (dropped == target_count)
            if correct:
                result_message = "Congratulations!"
                result_color = (0, 255, 0)
                score += 10
                correct_problems += 1
                if audio_loaded and congrats_sound:
                    print("Playing congratulation sound")
                    play_sound(congrats_sound)
                    current_sound = congrats_sound
            else:
                result_message = f"Try Again! {dropped} ≠ {target_count}"
                result_color = (255, 0, 0)
                if audio_loaded and tryagain_sound:
                    print("Playing try again sound")
                    play_sound(tryagain_sound)
                    current_sound = tryagain_sound
            
            # Log the interaction
            log_interaction(
                player_name, 
                player_age, 
                student_grade,
                GAME_MODE,
                problem_text, 
                dropped, 
                target_count,
                reaction_time, 
                correct, 
                score, 
                session_id, 
                TOTAL_SCREEN_TIME
            )
            
            show_result = True
        
        submit_pinched = False
    
    # Draw apples that are not in basket
    for apple in apples:
        if not apple["in_basket"]:
            screen.blit(apple_surface, (apple["x"], apple["y"]))
            
            # Visual feedback if apple has been counted
            if apple["has_been_counted"] and not apple["picked"]:
                # Draw a subtle glow effect
                glow_radius = APPLE_SIZE // 2 + 10
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (255, 255, 100, 100), (glow_radius, glow_radius), glow_radius)
                screen.blit(glow_surf, (apple["x"] + APPLE_SIZE//2 - glow_radius, 
                                      apple["y"] + APPLE_SIZE//2 - glow_radius))
    
    # Draw basket with appropriate number of apples
    basket_index = min(dropped, 5)
    current_basket_img = basket_images[basket_index]
    screen.blit(current_basket_img, (basket["x"], basket["y"]))
    
    # Draw submit button with 3D effect
    submit_color = (0, 180, 0)
    highlight_color = (0, 220, 0)
    shadow_color = (0, 140, 0)
    
    # Button shadow
    pygame.draw.rect(screen, shadow_color, 
                    (submit["x"] + 5, submit["y"] + 5, submit["w"], submit["h"]))
    
    # Button main
    pygame.draw.rect(screen, submit_color, 
                    (submit["x"], submit["y"], submit["w"], submit["h"]))
    
    # Button highlight
    pygame.draw.rect(screen, highlight_color, 
                    (submit["x"], submit["y"], submit["w"], 10))
    
    # Highlight border if hand over
    if submit_highlight:
        pygame.draw.rect(screen, (255, 255, 0), (submit["x"], submit["y"], submit["w"], submit["h"]), 3)
    
    submit_text = font.render("SUBMIT", True, (255, 255, 255))
    screen.blit(submit_text, (submit["x"] + 40, submit["y"] + 30))
    
    # Draw problem differently for COUNTING vs ADDITION modes
    if GAME_MODE == "COUNTING":
        # For COUNTING mode: Display just the large number in the center
        number_text = number_font.render(str(target_count), True, (255, 255, 200))
        number_rect = number_text.get_rect(center=(WIDTH // 2, 100))
        screen.blit(number_text, number_rect)
        
        # Draw small apples visualization - USING APPLE IMAGES INSTEAD OF CIRCLES
        apple_y = 200
        start_x = (WIDTH - (target_count * 90)) // 2
        
        for i in range(target_count):
            apple_x = start_x + i * 90
            apple_rect = small_apple_surface.get_rect(center=(apple_x, apple_y))
            if i < dropped:
                # Apple is in basket (filled)
                screen.blit(small_apple_surface, apple_rect)
                
                # Draw number on apple for better counting visualization
                if i < 9:  # Only draw numbers 1-9
                    num_text = small_font.render(str(i+1), True, (255, 255, 255))
                    num_rect = num_text.get_rect(center=(apple_x, apple_y))
                    screen.blit(num_text, num_rect)
            else:
                # Apple not yet in basket (semi-transparent)
                screen.blit(empty_apple_surface, apple_rect)
    
    else:  # ADDITION mode - NO RECTANGULAR BOX, USING APPLE IMAGES LIKE COUNTING
        # Draw problem WITHOUT background box - centered
        problem_display = large_font.render(problem_text, True, (255, 200, 150))
        problem_rect = problem_display.get_rect(center=(WIDTH // 2, 100))
        screen.blit(problem_display, problem_rect)
        
        # Draw apple images for ADDITION mode (same as COUNTING mode)
        apple_y = 200
        start_x = (WIDTH - (target_count * 90)) // 2
        
        for i in range(target_count):
            apple_x = start_x + i * 90
            apple_rect = small_apple_surface.get_rect(center=(apple_x, apple_y))
            if i < dropped:
                # Apple is in basket (filled)
                screen.blit(small_apple_surface, apple_rect)
            else:
                # Apple not yet in basket (semi-transparent)
                screen.blit(empty_apple_surface, apple_rect)
    
    # Draw score - position differently based on mode
    if GAME_MODE == "COUNTING":
        score_bg_x = WIDTH - 290  # Right side for COUNTING
    else:
        score_bg_x = 40  # Left side for ADDITION

    score_bg = pygame.Surface((250, 60), pygame.SRCALPHA)
    score_bg.fill((0, 0, 0, 150))
    screen.blit(score_bg, (score_bg_x, 120))
    
    score_text = font.render(f"Score: {score}", True, (255, 255, 0))
    screen.blit(score_text, (score_bg_x + 10, 130))
    
    # Draw current counting status (visual feedback for audio) - Only for COUNTING mode
    if GAME_MODE == "COUNTING" and audio_loaded and dropped > 0 and dropped <= 9:
        count_status_bg = pygame.Surface((200, 40), pygame.SRCALPHA)
        count_status_bg.fill((0, 0, 0, 150))
        screen.blit(count_status_bg, (WIDTH - 210, 200))
        
        count_status = small_font.render(f"Count: {dropped}", True, (100, 255, 100))
        screen.blit(count_status, (WIDTH - 200, 210))
    
    # Draw player info
    info_text = small_font.render(
        f"Player: {player_name[:10]} | Grade: {student_grade[:15]} | Time: {TOTAL_SCREEN_TIME:.0f}s", 
        True, (200, 200, 255)
    )
    screen.blit(info_text, (WIDTH - info_text.get_width() - 20, HEIGHT - 40))
    
    # Draw instructions with smaller font
    if GAME_MODE == "COUNTING":
        instruct_text = tiny_font.render("Pinch apples and drag to basket. Listen for number when dropped. Pinch SUBMIT when done.", True, (255, 255, 255))
    else:  # ADDITION
        instruct_text = tiny_font.render("Pinch apples and drag to basket. Pinch SUBMIT when done.", True, (255, 255, 255))
    screen.blit(instruct_text, (30, HEIGHT - 50))
    
    # Draw audio status indicator
    audio_status = "🔊 ON" if audio_loaded else "🔇 OFF"
    audio_text = tiny_font.render(f"Audio: {audio_status}", True, (100, 255, 100) if audio_loaded else (255, 100, 100))
    screen.blit(audio_text, (30, HEIGHT - 80))
    
    # Draw message with background
    if message:
        message_surf = font.render(message, True, (255, 0, 0))
        msg_width = message_surf.get_width()
        msg_height = message_surf.get_height()
        
        # Message background
        msg_bg = pygame.Surface((msg_width + 40, msg_height + 20), pygame.SRCALPHA)
        msg_bg.fill((255, 255, 255, 200))
        pygame.draw.rect(msg_bg, (0, 0, 255), (0, 0, msg_width + 40, msg_height + 20), 2)
        
        screen.blit(msg_bg, ((WIDTH - msg_width - 40) // 2, HEIGHT // 2 - 50))
        screen.blit(message_surf, ((WIDTH - msg_width) // 2, HEIGHT // 2 - 40))
    
    # Result screen
    if show_result:
        # Semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        # Result message
        result_surf = large_font.render(result_message, True, result_color)
        result_rect = result_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        screen.blit(result_surf, result_rect)
        
        # Score and accuracy
        if total_problems > 0:
            accuracy = (correct_problems / total_problems) * 100
            #stats_text = font.render(f"Score: {score}%", True, (255, 255, 200))
            #stats_rect = stats_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 20))
            #screen.blit(stats_text, stats_rect)
        
        # Continue instruction
        continue_surf = font.render("Pinch anywhere to continue", True, (255, 255, 255))
        continue_rect = continue_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 80))
        screen.blit(continue_surf, continue_rect)
    
    pygame.display.update()
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif show_result and event.key == pygame.K_SPACE:
                show_result = False
                reset_game()
                if GAME_MODE == "COUNTING":
                    target_count, problem_text = new_counting_question()
                else:  # ADDITION
                    target_count, problem_text, num1, num2 = new_addition_question()
                problem_start_time = time.time()
                if current_sound:
                    current_sound.stop()
                    current_sound = None
            elif event.key == pygame.K_m:
                # Return to game selection (for debugging)
                print("Returning to game selection...")
                running = False
            # Test counting sounds with number keys
            elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5,
                              pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                num = event.key - pygame.K_1 + 1
                if audio_loaded and num in counting_sounds and counting_sounds[num] is not None:
                    print(f"Testing sound for number {num}")
                    play_sound(counting_sounds[num])
            elif event.key == pygame.K_a:
                # Test all counting sounds
                if audio_loaded:
                    print("Testing all counting sounds...")
                    for i in range(1, 10):
                        if counting_sounds.get(i):
                            print(f"  Playing sound for {i}")
                            play_sound(counting_sounds[i])
                            pygame.time.delay(500)

# -----------------------------
# CLEANUP AND SESSION LOGGING
# -----------------------------
print("\n" + "="*50)
print("GAME SESSION SUMMARY")
print("="*50)

# Calculate average reaction time
average_reaction_time = sum(reaction_times) / len(reaction_times) if reaction_times else None

# Log session end
log_session_end(
    player_name, 
    player_age, 
    student_grade,
    GAME_MODE,
    session_start, 
    score, 
    total_problems, 
    correct_problems, 
    session_id,
    TOTAL_SCREEN_TIME,
    average_reaction_time
)

print(f"  Game Mode: {GAME_MODE}")
print(f"  Player: {player_name}")
print(f"  Final Score: {score}")
print(f"  Problems Solved: {total_problems}")
print(f"  Correct Answers: {correct_problems}")
if total_problems > 0:
    accuracy = (correct_problems / total_problems) * 100
    print(f"  Accuracy: {accuracy:.1f}%")
else:
    print(f"  Accuracy: N/A")
print(f"  Session Duration: {TOTAL_SCREEN_TIME:.1f} seconds")
if average_reaction_time:
    print(f"  Average Reaction Time: {average_reaction_time:.2f} seconds")
print(f"  Audio System: {'Enabled' if audio_loaded else 'Disabled'}")
print(f"  Audio Feature: Counting sounds play when apples are dropped in basket")
print("="*50)

# Cleanup
cap.release()
pygame.quit()
sys.exit(0)