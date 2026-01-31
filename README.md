# MOONDREAM

**Edge-Optimized AI Surveillance & Reasoning System**

Moondream is a next-generation AI surveillance assistant that combines classical computer vision with a local multimodal LLM to deliver explainable, evidence-driven security analysis on resource-constrained edge devices (e.g., 4GB VRAM GPUs).

This project is designed to be **hackathon-ready**, **police-relevant**, and **ethically grounded**, focusing on transparency, performance, and trust.

---

## Table of Contents

- [Why Moondream?](#why-moondream)
- [Architecture](#architecture)
- [Core Features](#core-features)
- [Local LLM Integration](#local-llm-integration)
- [Performance Philosophy](#performance-philosophy)
- [Event-Triggered Segmentation](#event-triggered-segmentation)
- [Installation](#installation)
- [Running](#running)
- [Output Example](#output-example)
- [Police & Public Safety Value](#police--public-safety-value)
- [Ethical & Legal Notes](#ethical--legal-notes)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why Moondream?

Most surveillance demos either:
- Run heavy deep models on every frame (slow, wasteful), or
- Rely on black-box AI decisions (hard to trust)

**Moondream is different.**

It follows a three-layer hybrid architecture that separates perception, evidence collection, and reasoning for maximum efficiency and explainability.

### Architecture

```mermaid
graph TB
    A["CAMERA INPUT"] --> B["LAYER 1: PERCEPTION<br/>FAST, DETERMINISTIC"]
    
    B --> B1["Motion Detection<br/>MOG2 + Frame Diff"]
    B --> B2["Person Detection<br/>HOG / YOLO-lite"]
    B --> B3["Face Detection<br/>Haar Cascades"]
    B --> B4["Tracking<br/>Centroid + Distance"]
    
    B1 --> C["LAYER 2: EVIDENCE & METRICS<br/>EXPLAINABLE"]
    B2 --> C
    B3 --> C
    B4 --> C
    
    C --> C1["People Count"]
    C --> C2["Motion Intensity"]
    C --> C3["Loitering Duration"]
    C --> C4["Movement Paths"]
    C --> C5["Heatmaps & Hotspots"]
    
    C1 --> D["LAYER 3: REASONING<br/>LOCAL LLM - MOONDREAM"]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    
    D --> D1["Threat Explanation"]
    D --> D2["Activity Classification"]
    D --> D3["Police Report"]
    D --> D4["Recommendations"]
    
    style A fill:#4a90e2,stroke:#2e5c8a,stroke-width:2px,color:#fff
    style B fill:#50c878,stroke:#2d7a4a,stroke-width:2px,color:#fff
    style C fill:#f39c12,stroke:#a67c00,stroke-width:2px,color:#fff
    style D fill:#e74c3c,stroke:#a93226,stroke-width:2px,color:#fff
    
    style B1 fill:#7bed9f,stroke:#2d7a4a,color:#000
    style B2 fill:#7bed9f,stroke:#2d7a4a,color:#000
    style B3 fill:#7bed9f,stroke:#2d7a4a,color:#000
    style B4 fill:#7bed9f,stroke:#2d7a4a,color:#000
    
    style C1 fill:#feca57,stroke:#a67c00,color:#000
    style C2 fill:#feca57,stroke:#a67c00,color:#000
    style C3 fill:#feca57,stroke:#a67c00,color:#000
    style C4 fill:#feca57,stroke:#a67c00,color:#000
    style C5 fill:#feca57,stroke:#a67c00,color:#000
    
    style D1 fill:#ff7979,stroke:#a93226,color:#000
    style D2 fill:#ff7979,stroke:#a93226,color:#000
    style D3 fill:#ff7979,stroke:#a93226,color:#000
    style D4 fill:#ff7979,stroke:#a93226,color:#000
```

> **Key Principle:** The LLM never replaces perception — it only explains evidence.

---

## Core Features

- ✅ Real-time motion detection
- ✅ Person & face detection (no identity recognition)
- ✅ Multi-person tracking & movement statistics
- ✅ Motion heatmaps & hotspot analysis
- ✅ Event-triggered LLM reasoning (local, offline)
- ✅ Structured threat assessment & reports
- ✅ Evidence packages (images + JSON)
- ✅ Optimized for RTX 3050 Mobile (4GB VRAM)

---

## Local LLM Integration

Moondream uses a **local multimodal LLM** via Ollama.

```mermaid
flowchart TD
    A["Camera Frame<br/>Original Resolution"] --> B{"Event<br/>Triggered?"}
    
    B -->|"No Activity"| A1["Skip Frame<br/>Save Resources"]
    B -->|"Motion/Person<br/>Detected"| C["Resize to 640x640<br/>Optimize for LLM"]
    
    C --> D["Base64 Encode<br/>Image Serialization"]
    
    D --> E["Prepare Prompt<br/>Evidence + Context"]
    
    E --> E1["Add People Count"]
    E --> E2["Add Motion Stats"]
    E --> E3["Add Tracking Data"]
    
    E1 --> F["Ollama API<br/>/api/generate"]
    E2 --> F
    E3 --> F
    
    F --> G["Moondream Model<br/>Local Inference"]
    
    G --> H["Structured Analysis"]
    
    H --> I1["Threat Level"]
    H --> I2["Activity Type"]
    H --> I3["Report"]
    H --> I4["Recommendations"]
    
    I1 --> J["Save Evidence Package<br/>JSON + Images"]
    I2 --> J
    I3 --> J
    I4 --> J
    
    J --> K["Output to Console<br/>& Dashboard"]
    
    style A fill:#4a90e2,stroke:#2e5c8a,stroke-width:2px,color:#fff
    style B fill:#9b59b6,stroke:#6c3483,stroke-width:2px,color:#fff
    style C fill:#3498db,stroke:#21618c,stroke-width:2px,color:#fff
    style D fill:#3498db,stroke:#21618c,stroke-width:2px,color:#fff
    style E fill:#1abc9c,stroke:#117a65,stroke-width:2px,color:#fff
    style F fill:#e67e22,stroke:#a04000,stroke-width:2px,color:#fff
    style G fill:#e74c3c,stroke:#a93226,stroke-width:2px,color:#fff
    style H fill:#f39c12,stroke:#a67c00,stroke-width:2px,color:#fff
    style J fill:#27ae60,stroke:#1e8449,stroke-width:2px,color:#fff
    style K fill:#2ecc71,stroke:#1e8449,stroke-width:2px,color:#fff
    style A1 fill:#95a5a6,stroke:#5d6d7e,stroke-width:2px,color:#fff
    
    style E1 fill:#48c9b0,stroke:#117a65,color:#000
    style E2 fill:#48c9b0,stroke:#117a65,color:#000
    style E3 fill:#48c9b0,stroke:#117a65,color:#000
    
    style I1 fill:#feca57,stroke:#a67c00,color:#000
    style I2 fill:#feca57,stroke:#a67c00,color:#000
    style I3 fill:#feca57,stroke:#a67c00,color:#000
    style I4 fill:#feca57,stroke:#a67c00,color:#000
```

**Privacy-First Design:**
- ✅ No cloud calls
- ✅ No OpenAI / external APIs
- ✅ Fully offline
- ✅ Privacy-preserving

---

## Performance Philosophy

Moondream is **event-driven**, not frame-driven.

| Approach | Description |
|----------|-------------|
| Vision Processing | Runs continuously (cheap) |
| AI Inference | Runs only when needed (expensive) |
| Frame Selection | Best frame selected for analysis |
| Model Triggering | Heavy models selectively triggered |

**Benefits:**
- Faster response times
- Higher accuracy
- Better scalability
- Suitable for edge deployment

---

## Event-Triggered Segmentation

To improve explainability and evidence quality, optional segmentation can be enabled:

```
IF (people_detected > 0 AND threat_level >= SUSPICIOUS):
    → Run SAM on selected frame
    → Extract pixel-accurate regions
    → Store masks as evidence
```

**Benefits:**
- Clear visual proof for police
- Strong hackathon differentiator
- No performance waste on normal scenes

---

## Installation

### Prerequisites

- Python 3.8+
- 4GB+ VRAM GPU (recommended)
- Ollama installed

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/obiwankenobi699/moondream.git
cd moondream
```

2. **Create virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install Ollama and pull model**

```bash
ollama pull moondream
```

---

## Running

Start the surveillance system:

```bash
uv run python EyerisAI.py
```

The system runs in **interactive surveillance sessions** and outputs:

- Real-time logs
- Evidence images
- Heatmaps
- JSON threat reports

---

## Output Example

```json
{
  "threat_level": "safe",
  "people_detected_max": 2,
  "faces_detected": 1,
  "motion_events": 32,
  "motion_intensity": "low",
  "recommended_action": "Continue routine monitoring",
  "ai_model_used": "moondream"
}
```

---

## Police & Public Safety Value

Moondream is designed to **assist — not replace** — human judgment.

### Key Advantages

- **Explainable Decisions**: Every assessment backed by evidence
- **Evidence-Based Reasoning**: No black-box predictions
- **Privacy-Aware Design**: No biometric identification
- **Clear Audit Trail**: Complete logging of all detections

### Ideal Use Cases

- Smart city surveillance
- Campus security
- Public transport monitoring
- Hackathons & research demos

---

## Ethical & Legal Notes

> **Important:** Always ensure compliance with local surveillance laws.

**Privacy Safeguards:**
- No face recognition or identity matching
- No continuous recording
- Event-based capture only
- Human-in-the-loop by design

---

## Roadmap

- [ ] Multi-camera orchestration
- [ ] Threat score with factor weights
- [ ] Officer-friendly dashboard
- [ ] Evidence hashing & chain-of-custody
- [ ] Real-time alert integrations

---

## License

MIT License

---

<p align="center">
  <sub>Built with ❤️ for safer communities</sub>
</p>
