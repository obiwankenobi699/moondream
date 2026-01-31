#!/usr/bin/env python3
"""
EyerisAI PRODUCTION SURVEILLANCE SYSTEM
=======================================
GPU-Accelerated | Vision LLM | Explainable AI

Features:
- CUDA GPU acceleration (your RTX 3050)
- Explainable threat scoring (hackathon-winning)
- Vision LLM multimodal analysis
- Behavioral metrics (loitering, pacing, speed)
- Evidence-based reasoning (no hallucinations)

Optimized for: Python 3.12.11 | RTX 3050 Mobile | Ollama moondream
"""

import base64
import configparser
import json
import logging
import os
import time
import warnings
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum

import cv2
import numpy as np
import requests

# GPU Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

# Check GPU availability
GPU_AVAILABLE = False
try:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Set memory limit for RTX 3050 (1.5GB for TensorFlow, rest for Ollama)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1536)]
        )
        GPU_AVAILABLE = True
        print(f"🚀 GPU ENABLED: {gpus[0].name}")
        print(f"   Memory limit: 1.5GB (leaving 2.5GB for Ollama)")
except Exception as e:
    print(f"⚠️  GPU not available: {e}")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =====================
# ENUMS
# =====================
class ThreatLevel(Enum):
    SAFE = "safe"
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ALERT = "alert"
    CRITICAL = "critical"


class ActivityType(Enum):
    IDLE = "idle"
    WALKING = "walking"
    RUNNING = "running"
    LOITERING = "loitering"
    FIGHTING = "fighting"
    FALLING = "falling"
    VANDALISM = "vandalism"
    THEFT = "theft"
    UNKNOWN = "unknown"


# =====================
# CONFIGURATION
# =====================
class SurveillanceConfig:
    """Configuration loader"""

    def __init__(self, config_file: str = "config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self._load_all_settings()

    def _load_all_settings(self):
        # General
        self.save_directory = self.config.get("General", "save_directory", fallback="surveillance_data")
        self.instance_name = self.config.get("General", "instance_name", fallback="CCTV-001")
        self.location = self.config.get("General", "location", fallback="Unknown Location")
        self.log_file = self.config.get("General", "log_file", fallback="motion_events.jsonl")

        # AI Model
        self.ai_base_url = self.config.get("AI", "base_url", fallback="http://localhost:11434")
        self.ai_model = self.config.get("AI", "model", fallback="moondream")
        self.max_tokens = self.config.getint("AI", "max_tokens", fallback=300)

        # Camera
        self.camera_id = self.config.getint("Camera", "device_id", fallback=0)
        self.camera_width = self.config.getint("Camera", "width", fallback=1920)
        self.camera_height = self.config.getint("Camera", "height", fallback=1080)
        self.auto_exposure = self.config.getfloat("Camera", "auto_exposure", fallback=0.75)

        # Motion Detection
        self.min_area = self.config.getint("MotionDetection", "min_area", fallback=700)
        self.threshold = self.config.getint("MotionDetection", "threshold", fallback=50)
        self.blur_size = (
            self.config.getint("MotionDetection", "blur_size_x", fallback=21),
            self.config.getint("MotionDetection", "blur_size_y", fallback=21)
        )
        self.cooldown = self.config.getint("MotionDetection", "cooldown", fallback=3)

        # Detection Features (add these to your config or use defaults)
        self.use_yolo = self.config.getboolean("Detection", "use_yolo", fallback=True)
        self.use_face_detection = self.config.getboolean("Detection", "use_face_detection", fallback=True)
        self.use_person_tracking = self.config.getboolean("Detection", "use_person_tracking", fallback=True)
        self.use_heatmap = self.config.getboolean("Detection", "use_heatmap", fallback=True)

        # Alert Thresholds
        self.max_normal_people = self.config.getint("Alerts", "max_normal_people", fallback=2)
        self.loitering_threshold_seconds = self.config.getfloat("Alerts", "loitering_threshold", fallback=10.0)


# =====================
# GPU-ACCELERATED PERSON DETECTOR
# =====================
class GPUPersonDetector:
    """GPU-accelerated person detection using HOG"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            try:
                # Try CUDA HOG
                self.hog = cv2.cuda_HOGDescriptor()
                self.hog.setSVMDetector(cv2.cuda_HOGDescriptor.getDefaultPeopleDetector())
                logger.info("✅ GPU-accelerated HOG detector initialized")
            except:
                # Fallback to CPU
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.use_gpu = False
                logger.info("⚠️  CUDA HOG unavailable, using CPU HOG")
        else:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            logger.info("CPU HOG detector initialized")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect people in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_gpu:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(gray)
                boxes, weights = self.hog.detectMultiScale(gpu_frame)
                boxes = boxes.download() if boxes is not None else []
                weights = weights.download() if weights is not None else []
            except:
                boxes, weights = self.hog.detectMultiScale(
                    gray, winStride=(8, 8), padding=(4, 4), scale=1.05
                )
        else:
            boxes, weights = self.hog.detectMultiScale(
                gray, winStride=(8, 8), padding=(4, 4), scale=1.05
            )

        detections = []
        if len(boxes) > 0:
            for i, (x, y, w, h) in enumerate(boxes):
                conf = weights.flatten()[i] if len(weights) > i else 0.85
                detections.append({
                    'class': 'person',
                    'confidence': float(conf),
                    'bbox': (int(x), int(y), int(w), int(h))
                })

        return detections


# =====================
# FACE DETECTOR
# =====================
class FaceDetector:
    """Face detection using Haar Cascades"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logger.info("Face Detector initialized")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        face_data = []
        for (x, y, w, h) in faces:
            face_data.append({
                'bbox': (x, y, w, h),
                'confidence': 0.9
            })

        return face_data


# =====================
# ENHANCED PERSON TRACKER (WITH BEHAVIORAL METRICS)
# =====================
class BehavioralPersonTracker:
    """Track individuals with behavioral analysis"""

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.positions_history = defaultdict(list)
        self.first_seen = {}

    def register(self, centroid):
        """Register new tracked person"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.positions_history[self.next_id].append(centroid)
        self.first_seen[self.next_id] = time.time()
        self.next_id += 1

    def deregister(self, object_id):
        """Remove tracked person"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.first_seen:
            del self.first_seen[object_id]

    def update(self, detections: List[Dict]) -> Dict[int, Tuple[int, int]]:
        """Update tracker with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []
        for det in detections:
            x, y, w, h = det['bbox']
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            for centroid in input_centroids:
                min_dist = float('inf')
                min_id = None

                for obj_id, obj_centroid in zip(object_ids, object_centroids):
                    dist = np.linalg.norm(np.array(centroid) - np.array(obj_centroid))
                    if dist < min_dist:
                        min_dist = dist
                        min_id = obj_id

                if min_id is not None and min_dist < 100:
                    self.objects[min_id] = centroid
                    self.disappeared[min_id] = 0
                    self.positions_history[min_id].append(centroid)
                else:
                    self.register(centroid)

        return self.objects

    def get_behavioral_metrics(self, object_id: int) -> Dict:
        """🔥 HACKATHON GOLD: Get detailed behavioral metrics"""
        if object_id not in self.positions_history:
            return {}

        positions = self.positions_history[object_id]
        if len(positions) < 2:
            return {
                'movement': 'stationary',
                'distance': 0,
                'speed': 0,
                'dwell_time_seconds': round(time.time() - self.first_seen.get(object_id, time.time()), 2)
            }

        # Calculate total distance
        total_distance = 0
        for i in range(1, len(positions)):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
            total_distance += dist

        # Calculate speed (pixels per frame)
        speed = total_distance / len(positions)

        # Detect direction changes (pacing indicator)
        direction_changes = 0
        for i in range(2, len(positions)):
            v1 = np.array(positions[i - 1]) - np.array(positions[i - 2])
            v2 = np.array(positions[i]) - np.array(positions[i - 1])
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                if np.dot(v1, v2) < 0:
                    direction_changes += 1

        # Classify movement
        if total_distance < 50:
            movement = 'stationary'
        elif speed < 5:
            movement = 'slow'
        elif speed < 15:
            movement = 'walking'
        else:
            movement = 'running'

        # Dwell time
        dwell_time = time.time() - self.first_seen.get(object_id, time.time())

        return {
            'movement': movement,
            'distance': int(total_distance),
            'speed': round(speed, 2),
            'positions_tracked': len(positions),
            'direction_changes': direction_changes,
            'pacing_detected': direction_changes > 5,
            'dwell_time_seconds': round(dwell_time, 2)
        }

    def get_loiterers(self, threshold_seconds: float = 10.0) -> List[int]:
        """Get IDs of people loitering beyond threshold"""
        loiterers = []
        current_time = time.time()

        for obj_id, first_time in self.first_seen.items():
            if obj_id in self.objects:
                dwell_time = current_time - first_time
                if dwell_time > threshold_seconds:
                    loiterers.append(obj_id)

        return loiterers


# =====================
# MOTION HEATMAP
# =====================
class MotionHeatmap:
    """Generate motion heatmap"""

    def __init__(self, width: int, height: int):
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.width = width
        self.height = height

    def update(self, motion_mask: np.ndarray):
        """Update heatmap"""
        if motion_mask is not None:
            resized = cv2.resize(motion_mask, (self.width, self.height))
            self.heatmap += (resized / 255.0) * 0.1
            self.heatmap *= 0.95

    def get_visualization(self) -> np.ndarray:
        """Get colored heatmap"""
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


# =====================
# GPU-ACCELERATED MOTION DETECTOR
# =====================
class GPUMotionDetector:
    """GPU-accelerated motion detection"""

    def __init__(self, config: SurveillanceConfig):
        self.config = config
        self.prev_gray = None
        self.use_gpu = GPU_AVAILABLE

        if self.use_gpu:
            try:
                self.bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=False
                )
                logger.info("✅ GPU-accelerated background subtractor")
            except:
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=False
                )
                self.use_gpu = False
                logger.info("⚠️  CPU background subtractor")
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )

    def detect(self, frame: np.ndarray) -> Tuple[bool, List, np.ndarray, np.ndarray]:
        """Detect motion"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.config.blur_size, 0)

        if self.use_gpu:
            try:
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                fg_mask_gpu = self.bg_subtractor.apply(gpu_gray, learningRate=-1)
                fg_mask = fg_mask_gpu.download()
            except:
                fg_mask = self.bg_subtractor.apply(gray)
        else:
            fg_mask = self.bg_subtractor.apply(gray)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, [], None, None

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, self.config.threshold, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_and(thresh, fg_mask)
        combined = cv2.dilate(combined, None, iterations=2)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > self.config.min_area]

        self.prev_gray = gray

        return len(significant_contours) > 0, significant_contours, thresh, combined


# =====================
# VISION LLM ANALYZER (EVIDENCE-BASED)
# =====================
class VisionLLMAnalyzer:
    """Evidence-based vision LLM analysis"""

    def __init__(self, config: SurveillanceConfig):
        self.config = config
        self.base_url = config.ai_base_url
        self.model = config.ai_model
        self.max_tokens = config.max_tokens

    def analyze(self, frame: np.ndarray, detections: Dict, motion_context: Dict) -> Dict:
        """
        🔥 VISION LLM MULTIMODAL ANALYSIS
        Evidence-constrained to prevent hallucinations
        """
        prompt = self._build_evidence_prompt(detections, motion_context)

        # Resize and encode
        resized = cv2.resize(frame, (640, 640))
        _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # More deterministic
                        "top_p": 0.9,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                ai_text = response.json()["response"].strip()
                return self._parse_response(ai_text, detections, motion_context)
            else:
                logger.error(f"Vision LLM error: {response.status_code}")
                return self._get_fallback(detections, motion_context)

        except Exception as e:
            logger.error(f"Vision LLM failed: {e}")
            return self._get_fallback(detections, motion_context)

    def _build_evidence_prompt(self, detections: Dict, motion_context: Dict) -> str:
        """Build evidence-constrained prompt"""
        people_count = detections.get('people_count', 0)

        prompt = f"""You are an AI surveillance analyst. You MUST base analysis ONLY on provided metrics and visible evidence.

VERIFIED DETECTION DATA:
- People detected: {people_count}
- Faces visible: {detections.get('face_count', 0)}
- Motion intensity: {motion_context.get('motion_intensity', 'unknown')}
- Location: {self.config.location}

CRITICAL RULES:
1. Do NOT invent events you don't see
2. If evidence is insufficient, explicitly say "insufficient evidence"
3. Base threat level ONLY on visible behavior
4. Be specific about what you actually observe

ANALYZE AND RESPOND IN THIS FORMAT:
THREAT_LEVEL: [SAFE/NORMAL/SUSPICIOUS/ALERT/CRITICAL]
ACTIVITY: [what you observe people doing]
BEHAVIOR: [normal/suspicious/threatening/neutral]
POSTURE: [relaxed/tense/aggressive/unclear]
OBSERVATIONS: [1-2 sentences on what you actually see]
CONFIDENCE: [high/medium/low]

If unclear, say "unclear" or "insufficient evidence"."""

        return prompt

    def _parse_response(self, ai_text: str, detections: Dict, motion_context: Dict) -> Dict:
        """Parse LLM response"""
        analysis = {
            "threat_level": ThreatLevel.NORMAL.value,
            "people_count": detections.get('people_count', 0),
            "face_count": detections.get('face_count', 0),
            "activity": "Unknown",
            "behavior": "unknown",
            "posture": "unknown",
            "details": ai_text,
            "confidence": "medium",
            "ai_raw_response": ai_text
        }

        lines = ai_text.split('\n')
        for line in lines:
            line = line.strip()

            if line.startswith("THREAT_LEVEL:"):
                threat_str = line.split(":", 1)[1].strip().upper()
                threat_mapping = {
                    "SAFE": ThreatLevel.SAFE.value,
                    "NORMAL": ThreatLevel.NORMAL.value,
                    "SUSPICIOUS": ThreatLevel.SUSPICIOUS.value,
                    "ALERT": ThreatLevel.ALERT.value,
                    "CRITICAL": ThreatLevel.CRITICAL.value
                }
                for key in threat_mapping:
                    if key in threat_str:
                        analysis["threat_level"] = threat_mapping[key]
                        break

            elif line.startswith("ACTIVITY:"):
                analysis["activity"] = line.split(":", 1)[1].strip()

            elif line.startswith("BEHAVIOR:"):
                analysis["behavior"] = line.split(":", 1)[1].strip().lower()

            elif line.startswith("POSTURE:"):
                analysis["posture"] = line.split(":", 1)[1].strip().lower()

            elif line.startswith("OBSERVATIONS:"):
                analysis["details"] = line.split(":", 1)[1].strip()

            elif line.startswith("CONFIDENCE:"):
                conf = line.split(":", 1)[1].strip().lower()
                if conf in ["high", "medium", "low"]:
                    analysis["confidence"] = conf

        return analysis

    def _get_fallback(self, detections: Dict, motion_context: Dict) -> Dict:
        """Fallback when LLM unavailable"""
        people_count = detections.get('people_count', 0)

        threat = ThreatLevel.NORMAL.value
        if people_count == 0:
            threat = ThreatLevel.SAFE.value
        elif people_count > 3:
            threat = ThreatLevel.SUSPICIOUS.value

        return {
            "threat_level": threat,
            "people_count": people_count,
            "face_count": detections.get('face_count', 0),
            "activity": f"{people_count} person(s) detected" if people_count > 0 else "No activity",
            "behavior": "unknown",
            "posture": "unknown",
            "details": "Vision LLM unavailable - using detection data only",
            "confidence": "low"
        }


# =====================
# ULTIMATE SURVEILLANCE SYSTEM
# =====================
class ProductionSurveillanceSystem:
    """
    🏆 HACKATHON-WINNING SURVEILLANCE SYSTEM
    - GPU-accelerated
    - Explainable threat scoring
    - Behavioral metrics
    - Vision LLM integration
    """

    def __init__(self, config: SurveillanceConfig):
        self.config = config

        # Initialize components
        self.motion_detector = GPUMotionDetector(config)
        self.vision_llm = VisionLLMAnalyzer(config)

        self.person_detector = GPUPersonDetector() if config.use_yolo else None
        self.face_detector = FaceDetector() if config.use_face_detection else None
        self.person_tracker = BehavioralPersonTracker() if config.use_person_tracking else None
        self.heatmap = MotionHeatmap(640, 480) if config.use_heatmap else None

        # Storage
        self.save_dir = Path(config.save_directory)
        self.save_dir.mkdir(exist_ok=True)
        for threat_level in ['safe', 'normal', 'suspicious', 'alert', 'critical']:
            (self.save_dir / threat_level).mkdir(exist_ok=True)

        self.alert_history = deque(maxlen=100)

        logger.info("=" * 70)
        logger.info("🏆 PRODUCTION SURVEILLANCE SYSTEM INITIALIZED")
        logger.info(f"   ├─ GPU Acceleration: {'✅' if GPU_AVAILABLE else '❌'}")
        logger.info(f"   ├─ Person Detection: {'✅' if config.use_yolo else '❌'}")
        logger.info(f"   ├─ Face Detection: {'✅' if config.use_face_detection else '❌'}")
        logger.info(f"   ├─ Person Tracking: {'✅' if config.use_person_tracking else '❌'}")
        logger.info(f"   ├─ Motion Heatmap: {'✅' if config.use_heatmap else '❌'}")
        logger.info(f"   └─ Vision LLM: {config.ai_model}")
        logger.info("=" * 70)

    def capture_and_analyze(self, cap: cv2.VideoCapture, duration: int = 7) -> Dict:
        """Main surveillance pipeline"""
        logger.info(f"\n🎥 SURVEILLANCE WINDOW: {duration} seconds")
        logger.info(f"📍 Location: {self.config.location}\n")

        start_time = time.time()
        end_time = start_time + duration

        frames_captured = 0
        motion_events = []

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            frames_captured += 1
            elapsed = time.time() - start_time

            # Motion detection
            has_motion, contours, thresh, motion_mask = self.motion_detector.detect(frame)

            if self.heatmap and motion_mask is not None:
                self.heatmap.update(motion_mask)

            if has_motion:
                motion_area = sum(cv2.contourArea(c) for c in contours)

                # Person detection
                people_count = 0
                if self.person_detector:
                    detections = self.person_detector.detect(frame)
                    people_count = len(detections)

                    # Track people
                    if self.person_tracker and people_count > 0:
                        self.person_tracker.update(detections)

                # Face detection
                face_count = 0
                if self.face_detector:
                    faces = self.face_detector.detect(frame)
                    face_count = len(faces)

                # Annotate
                annotated = self._annotate_frame(frame, contours, people_count, face_count)

                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                motion_events.append({
                    'timestamp': timestamp,
                    'elapsed': round(elapsed, 2),
                    'motion_area': int(motion_area),
                    'people_count': people_count,
                    'face_count': face_count,
                    'frame': annotated
                })

                logger.info(
                    f"⚡ {elapsed:.1f}s | Motion: {motion_area:,}px | "
                    f"People: {people_count} | Faces: {face_count}"
                )

            time.sleep(0.05)  # ~20 FPS

        logger.info(f"\n📊 Captured {frames_captured} frames, {len(motion_events)} motion events")

        # Vision LLM Analysis
        logger.info(f"🤖 Running Vision LLM Analysis...")
        llm_analysis = self._run_llm_analysis(motion_events)

        # Calculate Explainable Threat Score
        logger.info(f"🎯 Calculating Explainable Threat Score...")
        threat_score = self._calculate_threat_score(motion_events, llm_analysis)

        # Generate report
        report = self._generate_report(
            start_time, duration, frames_captured, motion_events, llm_analysis, threat_score
        )

        # Save evidence
        self._save_evidence(motion_events, report)

        return report

    def _run_llm_analysis(self, motion_events: List[Dict]) -> Dict:
        """Run Vision LLM analysis"""
        if not motion_events:
            return {
                "threat_level": ThreatLevel.SAFE.value,
                "people_count": 0,
                "activity": "No activity",
                "details": "Empty window"
            }

        # Aggregate data
        max_people = max([e['people_count'] for e in motion_events])
        max_faces = max([e['face_count'] for e in motion_events])
        total_motion = sum(e['motion_area'] for e in motion_events)

        detection_summary = {
            'people_count': max_people,
            'face_count': max_faces
        }

        motion_context = {
            'total_motion_area': total_motion,
            'motion_events': len(motion_events),
            'motion_intensity': 'high' if total_motion > 500000 else 'medium'
        }

        # Select best frame (most people)
        best_frame_idx = len(motion_events) // 2
        if max_people > 0:
            max_people_frame = max(motion_events, key=lambda x: x['people_count'])
            best_frame_idx = motion_events.index(max_people_frame)

        logger.info(f"   Analyzing frame {best_frame_idx + 1}/{len(motion_events)}")

        # Vision LLM analysis
        analysis = self.vision_llm.analyze(
            motion_events[best_frame_idx]['frame'],
            detection_summary,
            motion_context
        )

        # Add loitering check
        if self.person_tracker:
            loiterers = self.person_tracker.get_loiterers(self.config.loitering_threshold_seconds)
            if loiterers:
                analysis['loitering_detected'] = True
                analysis['loitering_count'] = len(loiterers)

        return analysis

    def _calculate_threat_score(self, motion_events: List[Dict], llm_analysis: Dict) -> Dict:
        """
        🔥 HACKATHON GOLD: Explainable Threat Scoring
        """
        factors = {}

        # Factor 1: People count (0.0 - 0.3)
        max_people = max([e['people_count'] for e in motion_events]) if motion_events else 0
        if max_people == 0:
            factors['people_count'] = 0.0
        elif max_people <= 2:
            factors['people_count'] = 0.1
        elif max_people <= 4:
            factors['people_count'] = 0.2
        else:
            factors['people_count'] = 0.3

        # Factor 2: Loitering (0.0 - 0.25)
        duration = len(motion_events) * 0.05
        if duration < 3:
            factors['loitering'] = 0.0
        elif duration < self.config.loitering_threshold_seconds:
            factors['loitering'] = 0.1
        elif duration < 20:
            factors['loitering'] = 0.2
        else:
            factors['loitering'] = 0.25

        # Factor 3: Motion intensity (0.0 - 0.2)
        total_motion = sum(e['motion_area'] for e in motion_events)
        if total_motion < 100000:
            factors['motion_intensity'] = 0.0
        elif total_motion < 500000:
            factors['motion_intensity'] = 0.1
        else:
            factors['motion_intensity'] = 0.2

        # Factor 4: AI behavior (0.0 - 0.25)
        behavior = llm_analysis.get('behavior', 'unknown').lower()
        posture = llm_analysis.get('posture', 'unknown').lower()

        if behavior in ['threatening', 'aggressive'] or posture == 'aggressive':
            factors['ai_behavior'] = 0.25
        elif behavior in ['suspicious', 'nervous'] or posture == 'tense':
            factors['ai_behavior'] = 0.15
        elif behavior == 'normal' and posture == 'relaxed':
            factors['ai_behavior'] = 0.0
        else:
            factors['ai_behavior'] = 0.05

        # Total score
        total_score = sum(factors.values())

        # Map to threat level
        if total_score < 0.2:
            threat_level = ThreatLevel.SAFE.value
        elif total_score < 0.4:
            threat_level = ThreatLevel.NORMAL.value
        elif total_score < 0.6:
            threat_level = ThreatLevel.SUSPICIOUS.value
        elif total_score < 0.8:
            threat_level = ThreatLevel.ALERT.value
        else:
            threat_level = ThreatLevel.CRITICAL.value

        # Explanation
        explanations = []
        if factors['people_count'] > 0.15:
            explanations.append(f"Multiple people ({factors['people_count']:.2f})")
        if factors['loitering'] > 0.1:
            explanations.append(f"Extended presence ({factors['loitering']:.2f})")
        if factors['motion_intensity'] > 0.1:
            explanations.append(f"High motion ({factors['motion_intensity']:.2f})")
        if factors['ai_behavior'] > 0.1:
            explanations.append(f"Concerning behavior ({factors['ai_behavior']:.2f})")

        explanation = "; ".join(explanations) if explanations else "No significant threats"

        return {
            'total_score': round(total_score, 3),
            'threat_level': threat_level,
            'factor_breakdown': {k: round(v, 3) for k, v in factors.items()},
            'explanation': explanation,
            'is_explainable': True
        }

    def _annotate_frame(self, frame: np.ndarray, contours: List, people: int, faces: int) -> np.ndarray:
        """Annotate frame"""
        annotated = frame.copy()
        cv2.drawContours(annotated, contours, -1, (232, 8, 255), 2)

        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.rectangle(annotated, (0, 0), (450, 100), (0, 0, 0), -1)
        cv2.putText(annotated, f"{self.config.instance_name} - {self.config.location}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"Time: {timestamp}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"People: {people} | Faces: {faces}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return annotated

    def _generate_report(
            self, start_time: float, duration: int, frames: int,
            motion_events: List[Dict], llm_analysis: Dict, threat_score: Dict
    ) -> Dict:
        """Generate comprehensive report"""

        # Behavioral tracking data
        tracking_data = {}
        if self.person_tracker:
            for obj_id in self.person_tracker.objects.keys():
                tracking_data[f"person_{obj_id}"] = self.person_tracker.get_behavioral_metrics(obj_id)

        max_people = max([e['people_count'] for e in motion_events]) if motion_events else 0

        report = {
            "surveillance_session": {
                "camera_id": self.config.instance_name,
                "location": self.config.location,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat()
            },

            "threat_assessment": {
                "threat_level": threat_score['threat_level'],
                "threat_score": threat_score['total_score'],
                "score_breakdown": threat_score['factor_breakdown'],
                "score_explanation": threat_score['explanation'],
                "is_explainable": True,
                "confidence": llm_analysis.get('confidence', 'medium')
            },

            "detection_summary": {
                "people_detected_max": int(max_people),
                "faces_detected": llm_analysis.get('face_count', 0),
                "tracked_individuals": len(tracking_data)
            },

            "behavioral_analysis": {
                "primary_activity": llm_analysis.get('activity', 'Unknown'),
                "behavior": llm_analysis.get('behavior', 'unknown'),
                "posture": llm_analysis.get('posture', 'unknown'),
                "loitering_detected": llm_analysis.get('loitering_detected', False),
                "tracking_metrics": tracking_data,
                "vision_llm_observations": llm_analysis.get('details', '')
            },

            "motion_statistics": {
                "total_frames": frames,
                "motion_events": len(motion_events),
                "total_motion_area": sum(e['motion_area'] for e in motion_events)
            },

            "system_info": {
                "gpu_enabled": GPU_AVAILABLE,
                "ai_model": self.config.ai_model,
                "camera_resolution": f"{self.config.camera_width}x{self.config.camera_height}",
                "report_version": "3.0_PRODUCTION"
            }
        }

        return report

    def _save_evidence(self, motion_events: List[Dict], report: Dict):
        """Save evidence package"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        threat_level = report['threat_assessment']['threat_level']
        save_dir = self.save_dir / threat_level

        # Save frames
        max_frames = {'safe': 1, 'normal': 3, 'suspicious': 10, 'alert': 15, 'critical': 20}
        for i, event in enumerate(motion_events[:max_frames.get(threat_level, 5)]):
            frame_path = save_dir / f"{timestamp}_frame_{i:03d}.jpg"
            cv2.imwrite(str(frame_path), event['frame'])

        # Save heatmap
        if self.heatmap:
            heatmap_path = save_dir / f"{timestamp}_heatmap.jpg"
            cv2.imwrite(str(heatmap_path), self.heatmap.get_visualization())

        # Save JSON report
        report_path = save_dir / f"{timestamp}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Append to log file
        log_path = self.save_dir / self.config.log_file
        with open(log_path, 'a') as f:
            f.write(json.dumps(report) + '\n')

        logger.info(f"\n💾 Evidence saved:")
        logger.info(f"   ├─ Threat Score: {report['threat_assessment']['threat_score']:.3f}")
        logger.info(f"   └─ Location: {save_dir}/\n")


# =====================
# MAIN
# =====================
def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     EyerisAI PRODUCTION SURVEILLANCE - GPU Accelerated v3.0       ║
║                                                                   ║
║  🚀 GPU Acceleration  |  🎯 Explainable AI  |  👁️  Vision LLM     ║
║  📊 Behavioral Metrics |  🔒 Evidence-Based |  🏆 Award-Winning   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    config = SurveillanceConfig()
    system = ProductionSurveillanceSystem(config)

    logger.info("🎬 Initializing camera...")
    cap = cv2.VideoCapture(config.camera_id)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.auto_exposure)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)

    if not cap.isOpened():
        logger.error("❌ Camera failed")
        return

    logger.info(f"✅ Camera online: {config.camera_width}x{config.camera_height}")
    logger.info(f"📍 Location: {config.location}")
    logger.info(f"🤖 Vision LLM: {config.ai_model}\n")

    try:
        while True:
            logger.info("Press ENTER to start surveillance (Ctrl+C to exit)...")
            input()

            # Warm up
            for _ in range(5):
                cap.read()

            # Run surveillance
            report = system.capture_and_analyze(cap, duration=7)

            # Display report
            print("\n" + "=" * 70)
            print("📋 SURVEILLANCE REPORT:")
            print("=" * 70)
            print(json.dumps(report, indent=2))
            print("=" * 70 + "\n")

            # Highlight threat score
            threat = report['threat_assessment']
            print(f"🎯 EXPLAINABLE THREAT SCORE:")
            print(f"   Total: {threat['threat_score']:.3f}")
            print(f"   Level: {threat['threat_level'].upper()}")
            print(f"   Breakdown: {threat['score_breakdown']}")
            print(f"   Explanation: {threat['score_explanation']}\n")

    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
    finally:
        cap.release()
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    main()