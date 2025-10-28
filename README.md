# Polytrack AI 🎮

A reinforcement learning agent trained to master the Polytrack video game and compete for world records.

## Overview

This project uses **Dueling Deep Q-Networks (Dueling DQN)** to train an autonomous agent that learns to play Polytrack optimally. The agent uses computer vision for real-time game state analysis and achieves superhuman performance through deep reinforcement learning.

## Features

- **Dueling DQN Architecture**: Advanced reinforcement learning model with separate value and advantage streams
- **Real-time Screen Capture**: Low-latency frame capture using `mss` library (~50ms per frame)
- **OCR-based Metrics Extraction**: Extracts speed and time data directly from game UI
- **Replay Buffer**: Experience replay for efficient learning from past gameplay
- **Model Checkpointing**: Automatic saving of best-performing models
- **PyAutoGUI Control**: Direct keyboard input without bot detection

## Installation

### Prerequisites
- Python 3.8+
- MacBook Air (or any machine with CUDA/MPS support)
- Polytrack game accessible in browser

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/polytrack-ai.git
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

## Usage

1. Open Polytrack in your browser and keep it visible
2. Run the training script:
```bash
python src/train.py
```

3. The agent will start learning. Monitor progress in the console output.
4. Best models are automatically saved to `models/polytrack_model.pt`

## Project Structure

```
polytrack-ai/
├── src/
│   ├── train.py           # Main training loop
│   ├── agent.py           # DQN agent implementation
│   ├── environment.py     # Game interaction & screen capture
│   └── utils.py           # Helper functions
├── config/
│   └── settings.py        # Configuration parameters
├── models/                # Saved model checkpoints
├── logs/                  # Training logs
├── data/                  # Replay buffer data
├── requirements.txt       # Python dependencies
└── README.md
```

## How It Works

### 1. Vision Processing
- Captures 50ms screenshots using `mss`
- CNN encoder extracts features from game state
- OCR reads speed/time metrics from UI

### 2. Decision Making
- Dueling DQN processes game state
- Selects actions: left, right, forward, idle
- Uses epsilon-greedy exploration strategy

### 3. Learning
- Stores experiences in replay buffer
- Trains on mini-batches from replay memory
- Updates target network periodically
- Saves best models based on episode rewards

## Results

| Metric | Value |
|--------|-------|
| Best Episode Reward | TBD |
| Training Episodes | TBD |
| Average Frame Processing | ~50ms |
| Model Size | ~2MB |

## Technologies Used

- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **EasyOCR**: Text extraction from UI
- **mss**: Low-latency screenshot capture
- **PyAutoGUI**: Game input control

## Future Improvements

- [ ] Implement prioritized experience replay
- [ ] Add A3C (Asynchronous Advantage Actor-Critic)
- [ ] Multi-agent training for comparison
- [ ] Web dashboard for training visualization
- [ ] Support for multiple game modes

## Contributing

Feel free to fork, modify, and improve! This is a learning project.

## License

MIT License - see LICENSE file for details

## Author

Saurya Aditya Sahu
