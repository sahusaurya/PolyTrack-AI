# Polytrack AI 🎮🤖

A reinforcement learning agent trained to master the Polytrack racing game and compete for world records using deep learning and computer vision.

## Overview

This project uses **Dueling Deep Q-Networks (Dueling DQN)** combined with real-time computer vision to train an autonomous agent that learns optimal racing strategies in Polytrack. The agent processes game frames directly, detects game state through pixel analysis, and learns to make split-second decisions to achieve superhuman performance.

## Features

- **Dueling DQN Architecture**: Advanced RL model with separate value and advantage streams for better decision-making
- **Real-time Vision Processing**: Low-latency screen capture at ~50ms per frame using `mss`
- **Intelligent Crash Detection**: Multi-method crash detection including sudden speed drops and stuck vehicle detection
- **Vision-Based Metrics**: Pixel-intensity analysis for speed tracking (no OCR required)
- **Experience Replay**: Efficient learning from past gameplay stored in replay buffer
- **Automatic Checkpointing**: Saves best-performing models automatically
- **MPS Optimization**: Leverages Apple Silicon M3 GPU acceleration via Metal Performance Shaders
- **Graceful Interruption**: Press Ctrl+C anytime to safely stop and resume training later

## Technical Highlights

### Why Vision-Based Instead of OCR?
Initially attempted OCR for reading speed/time metrics, but pixel-intensity analysis proved:
- **Faster**: No OCR processing overhead
- **More reliable**: No misreading of digits
- **Better for RL**: Agent learns from visual patterns rather than numeric values

### Crash Detection System
Implements dual crash detection:
1. **High-speed crashes**: Detects sudden speed drops (>30 units)
2. **Stuck detection**: Monitors speed variance over time to catch low-speed collisions and getting wedged on track edges

## Installation

### Prerequisites
- Python 3.8+
- MacBook Air M3 (or any machine with MPS/CUDA support)
- Polytrack accessible in browser
- macOS with Homebrew (for Tesseract, optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sahusaurya/polytrack-ai.git
cd polytrack-ai
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

4. (Optional) Install Tesseract for OCR experiments:
```bash
brew install tesseract
```

## Usage

1. Open Polytrack in your browser (Brave recommended) and go fullscreen (F11)
2. Run the training script:
```bash
python src/train.py
```

3. Press ENTER when prompted, then quickly switch to the Polytrack window (you have 3 seconds)
4. The AI takes over and starts learning!

### Stopping and Resuming Training
- Press `Ctrl+C` anytime to safely stop training
- Model is automatically saved
- Run `python src/train.py` again to resume from the last checkpoint

## Project Structure

```
polytrack-ai/
├── src/
│   ├── train.py              # Main training loop with DQN agent
│   ├── test_capture.py       # Test screen capture
│   ├── test_ocr.py           # Test OCR regions (legacy)
│   └── find_regions.py       # Helper to find UI element positions
├── models/
│   └── polytrack_best.pt     # Best model checkpoint (auto-saved)
├── logs/                     # Training logs (future)
├── data/                     # Replay buffer data (future)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
└── README.md
```

## How It Works

### 1. Vision System
- Captures 240x160 screenshots at 20 FPS using `mss`
- CNN encoder extracts spatial features from game state
- Pixel-intensity analysis tracks speed changes
- Hash-based timer monitoring detects race completion

### 2. Decision Making
- Dueling DQN processes visual features
- 5 possible actions: idle, forward (W), left (A), right (D), brake (S)
- Epsilon-greedy exploration (starts at 100%, decays to 1%)
- PyAutoGUI sends keyboard commands directly to game

### 3. Learning Process
- Experience replay buffer stores (state, action, reward, next_state)
- Batched training on mini-batches of 64 experiences
- Target network updated every 100 steps for stability
- Smooth L1 loss for robust Q-value learning

### 4. Reward Structure
- **+0.1** per step survived
- **+(speed/100) × 0.1** for maintaining speed
- **+100** for finishing the race
- **-10** for crashing

## Training Performance

| Metric | Value |
|--------|-------|
| Input Resolution | 240x160 RGB |
| Frame Processing | ~50ms per frame |
| Actions per Second | ~20 |
| Model Parameters | ~8.5M |
| Model Size | ~35MB |
| Device | Apple M3 (MPS) |
| Replay Buffer | 50,000 transitions |

## Technologies Used

- **PyTorch**: Deep learning framework with MPS acceleration
- **OpenCV**: Image processing and computer vision
- **mss**: Ultra-fast screenshot capture
- **PyAutoGUI**: Keyboard/mouse automation
- **NumPy**: Numerical computing
- **Tesseract** (optional): OCR experiments

## Development Journey

### Challenges Overcome
1. **Backend Access**: Initially tried accessing game backend - not possible with browser games
2. **Selenium Detection**: Game blocked automated browsers, switched to direct screen capture
3. **OCR Unreliability**: Tesseract struggled with game fonts, pivoted to pixel-intensity analysis
4. **MPS Compatibility**: Fixed adaptive pooling issues specific to Apple Silicon
5. **Stuck Detection**: Added sophisticated crash detection beyond simple speed monitoring

## Future Improvements

- [ ] Implement prioritized experience replay for faster learning
- [ ] Add curriculum learning (start with easier tracks)
- [ ] Multi-track support and track-specific models
- [ ] Real-time training visualization dashboard
- [ ] A3C (Asynchronous Advantage Actor-Critic) for parallel training
- [ ] Imitation learning from human gameplay recordings
- [ ] Transfer learning across similar racing games

## Contributing

This is a learning project! Feel free to:
- Fork and experiment
- Submit issues for bugs
- Suggest improvements via PRs
- Share your training results

## License

MIT License - See LICENSE file for details

## Author

**Saurya Aditya Sahu** ([@sahusaurya](https://github.com/sahusaurya))

- Inspired by DeepMind's game-playing agents (DQN paper, AlphaGo)
- Built with PyTorch and the RL community's best practices
- Thanks to Polytrack developers for creating an engaging test environment
