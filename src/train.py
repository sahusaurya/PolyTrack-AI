import cv2
import numpy as np
import mss
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import json
from pathlib import Path
from datetime import datetime
import re

# Disable PyAutoGUI failsafe for smooth operation
pyautogui.FAILSAFE = False

# ============ CONFIGURATION ============
CONFIG = {
    # Screen settings for 1470x956 fullscreen
    "game_window": {"top": 0, "left": 0, "width": 1470, "height": 956},
    
    # OCR regions for speed and time (exact coordinates from user)
    "speed_region": {"top": 900, "left": 1300, "width": 170, "height": 56},  # Bottom far right
    "time_region": {"top": 900, "left": 600, "width": 270, "height": 56},    # Bottom center
    "finish_popup_region": {"top": 378, "left": 585, "width": 300, "height": 200},  # Center screen
    
    # Training parameters
    "frame_skip": 2,                    # Process every 2nd frame for speed
    "screenshot_interval": 0.05,        # 50ms between actions
    "crash_speed_threshold": 30,        # Speed drop indicating crash
    "max_episode_steps": 1000,          # Max steps per episode
    
    # Model parameters
    "model_checkpoint": "models/polytrack_best.pt",
    "replay_buffer_size": 50000,
    "batch_size": 64,
    "learning_rate": 0.0001,
    "gamma": 0.99,                      # Discount factor
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.9995,
    "target_update_freq": 100,
    
    # Reward shaping
    "reward_time_weight": 10.0,         # Reward for fast completion
    "reward_speed_weight": 0.1,         # Reward for maintaining speed
    "reward_survival": 0.1,             # Small reward for each step alive
    "penalty_crash": -10.0,             # Penalty for crashing
}

# ============ VISION MODEL (Optimized for M3) ============
class GameStateEncoder(nn.Module):
    """Lightweight CNN for M3 Mac"""
    def __init__(self):
        super().__init__()
        # Smaller network for faster training on M3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate flattened size
        self.feature_size = 64 * 28 * 28
        
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return x

class DuelingDQN(nn.Module):
    """Dueling DQN for optimal action selection"""
    def __init__(self, num_actions=5):
        super().__init__()
        self.encoder = GameStateEncoder()
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        self.num_actions = num_actions
        
    def forward(self, x):
        features = self.encoder(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages
        qvalues = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvalues

# ============ SCREEN CAPTURE & METRICS ============
class PolytrackCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.game_region = CONFIG["game_window"]
        self.speed_region = CONFIG["speed_region"]
        self.time_region = CONFIG["time_region"]
        self.finish_popup_region = CONFIG["finish_popup_region"]
        
        self.current_speed = 0
        self.previous_speed = 0
        self.current_time = 0
        self.previous_time = 0
        self.time_stopped_count = 0
        
    def capture_game_frame(self):
        """Capture full game screen"""
        screenshot = self.sct.grab(self.game_region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        # Resize for neural network (smaller = faster)
        frame = cv2.resize(frame, (240, 160))
        return frame
    
    def extract_speed(self):
        """Extract speed using pixel intensity instead of OCR"""
        screenshot = self.sct.grab(self.speed_region)
        frame = np.array(screenshot)
        
        # Convert to grayscale and measure brightness
        # More bright pixels = higher number displayed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        
        # Count bright pixels (the digits)
        bright_pixels = np.sum(gray > 200)
        
        # Estimate speed based on bright pixel count
        # This is a rough approximation but works for detecting changes
        estimated_speed = bright_pixels / 10.0  # Normalize
        
        return estimated_speed
    
    def extract_time(self):
        """Extract time display state using pixel patterns"""
        screenshot = self.sct.grab(self.time_region)
        frame = np.array(screenshot)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        
        # Return hash of the image to detect changes
        return hash(gray.tobytes())
    
    def check_finish_line(self):
        """Check if race is finished by detecting timer stop or popup"""
        # Method 1: Check if timer display stopped changing
        current_time_hash = self.extract_time()
        
        if current_time_hash == self.previous_time and self.previous_time != 0:
            self.time_stopped_count += 1
            if self.time_stopped_count > 5:  # Timer hasn't changed for 5 checks
                return True
        else:
            self.time_stopped_count = 0
        
        self.previous_time = current_time_hash
        
        # Method 2: Check for popup in center screen
        screenshot = self.sct.grab(self.finish_popup_region)
        frame = np.array(screenshot)
        
        # Check if there's a bright popup
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        brightness = np.mean(gray)
        
        if brightness > 180:  # Bright popup detected
            return True
        
        return False
    
    def detect_crash(self):
        """Detect crash by monitoring speed drop"""
        self.previous_speed = self.current_speed
        self.current_speed = self.extract_speed()
        
        # Crash if speed drops dramatically without braking
        speed_drop = self.previous_speed - self.current_speed
        if speed_drop > CONFIG["crash_speed_threshold"] and self.previous_speed > 50:
            return True
        return False
    
    def get_state(self):
        """Get current game state"""
        frame = self.capture_game_frame()
        speed = self.extract_speed()
        crashed = self.detect_crash()
        finished = self.check_finish_line()
        
        return {
            'frame': frame,
            'speed': speed,
            'crashed': crashed,
            'finished': finished
        }

# ============ GAME CONTROLLER ============
class PolytrackController:
    """Control Polytrack using keyboard"""
    def __init__(self):
        # Action mapping: 0=nothing, 1=forward, 2=left, 3=right, 4=brake
        self.actions = {
            0: self.idle,
            1: self.forward,
            2: self.left,
            3: self.right,
            4: self.brake,
        }
        self.last_action = 0
        
    def idle(self):
        """Do nothing"""
        pass
    
    def forward(self):
        """Press W"""
        pyautogui.keyDown('w')
        time.sleep(0.05)
        pyautogui.keyUp('w')
    
    def left(self):
        """Press A"""
        pyautogui.keyDown('a')
        time.sleep(0.05)
        pyautogui.keyUp('a')
    
    def right(self):
        """Press D"""
        pyautogui.keyDown('d')
        time.sleep(0.05)
        pyautogui.keyUp('d')
    
    def brake(self):
        """Press S"""
        pyautogui.keyDown('s')
        time.sleep(0.05)
        pyautogui.keyUp('s')
    
    def restart_game(self):
        """Press T to restart"""
        pyautogui.press('t')
        time.sleep(0.3)  # Minimal delay for key registration
    
    def execute(self, action):
        """Execute an action"""
        self.actions[action]()
        self.last_action = action

# ============ REPLAY BUFFER ============
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2) / 255.0,
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2) / 255.0,
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# ============ TRAINING AGENT ============
class PolytrackAgent:
    def __init__(self):
        # Use MPS (Metal Performance Shaders) for M3 Mac
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Models
        self.policy_net = DuelingDQN(num_actions=5).to(self.device)
        self.target_net = DuelingDQN(num_actions=5).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=CONFIG["learning_rate"])
        
        # Components
        self.replay_buffer = ReplayBuffer(CONFIG["replay_buffer_size"])
        self.capture = PolytrackCapture()
        self.controller = PolytrackController()
        
        # Training state
        self.epsilon = CONFIG["epsilon_start"]
        self.steps_done = 0
        self.episode_count = 0
        self.best_reward = -float('inf')
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 5)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def compute_reward(self, state_info, action, crashed, finished):
        """Compute reward for current step"""
        reward = CONFIG["reward_survival"]  # Base survival reward
        
        # Reward for maintaining speed
        speed = state_info['speed']
        reward += (speed / 100.0) * CONFIG["reward_speed_weight"]
        
        # Big reward for finishing the race!
        if finished:
            reward += 100.0  # Huge bonus for completion
        
        # Penalize crash
        if crashed:
            reward += CONFIG["penalty_crash"]
        
        return reward
    
    def train_step(self):
        """Single training step"""
        if len(self.replay_buffer) < CONFIG["batch_size"]:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(CONFIG["batch_size"])
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + CONFIG["gamma"] * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def play_episode(self):
        """Play one full episode"""
        print(f"\n=== Episode {self.episode_count + 1} ===")
        
        # Restart game
        self.controller.restart_game()
        time.sleep(0.5)  # Brief pause for game reset
        
        # Initialize episode
        state_info = self.capture.get_state()
        state = state_info['frame']
        episode_reward = 0
        episode_loss = []
        
        for step in range(CONFIG["max_episode_steps"]):
            # Select and execute action
            action = self.select_action(state)
            self.controller.execute(action)
            
            # Wait and capture next state
            time.sleep(CONFIG["screenshot_interval"])
            next_state_info = self.capture.get_state()
            next_state = next_state_info['frame']
            
            # Compute reward
            crashed = next_state_info['crashed']
            finished = next_state_info['finished']
            reward = self.compute_reward(next_state_info, action, crashed, finished)
            episode_reward += reward
            
            # Store transition
            done = crashed or finished
            self.replay_buffer.push(state, action, reward, next_state, float(done))
            
            # Train
            loss = self.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update target network
            self.steps_done += 1
            if self.steps_done % CONFIG["target_update_freq"] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Check if episode ended
            if crashed:
                print(f"💥 Crashed at step {step}!")
                break
            
            if finished:
                print(f"🏁 FINISHED THE RACE at step {step}!")
                break
            
            state = next_state
            
            # Print progress
            if step % 50 == 0:
                print(f"Step {step} | Speed: {next_state_info['speed']:.1f} | Reward: {episode_reward:.2f}")
        
        # Update epsilon
        self.epsilon = max(CONFIG["epsilon_end"], self.epsilon * CONFIG["epsilon_decay"])
        
        # Save best model
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.save_model()
            print(f"🎉 New best reward: {self.best_reward:.2f}")
        
        self.episode_count += 1
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        
        return {
            'reward': episode_reward,
            'steps': step,
            'avg_loss': avg_loss,
            'epsilon': self.epsilon
        }
    
    def save_model(self):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode_count,
            'best_reward': self.best_reward
        }, CONFIG["model_checkpoint"])
        print(f"Model saved to {CONFIG['model_checkpoint']}")
    
    def load_model(self):
        """Load model checkpoint"""
        if Path(CONFIG["model_checkpoint"]).exists():
            checkpoint = torch.load(CONFIG["model_checkpoint"], map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.episode_count = checkpoint['episode']
            self.best_reward = checkpoint['best_reward']
            print(f"Model loaded! Best reward: {self.best_reward:.2f}")
            return True
        return False

# ============ MAIN ============
def main():
    print("=" * 60)
    print("POLYTRACK AI TRAINER")
    print("=" * 60)
    print("\nMake sure:")
    print("1. Polytrack is open in Brave browser (FULLSCREEN)")
    print("2. The game window is visible and focused")
    print("3. You're ready to let the AI take over!\n")
    
    input("Press ENTER when ready...")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Initialize agent
    agent = PolytrackAgent()
    agent.load_model()
    
    try:
        for episode in range(1000):
            results = agent.play_episode()
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1} Complete!")
            print(f"Reward: {results['reward']:.2f} | Best: {agent.best_reward:.2f}")
            print(f"Steps: {results['steps']} | Loss: {results['avg_loss']:.4f}")
            print(f"Epsilon: {results['epsilon']:.3f}")
            print(f"{'='*60}\n")
            
            # Short break between episodes
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user!")
        agent.save_model()
        print("Final model saved. Goodbye!")

if __name__ == "__main__":
    main()