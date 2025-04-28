# Teleoperation Maze Game with Q-Learning

A Python/Pygame simulation and reinforcement-learning framework where a mobile agent must **push** and **pull** a movable block through a procedurally generated maze to a goal. Supports both:

- **Manual teleoperation** via keyboard (WASD or arrow keys)  
- **Autonomous control** via tabular Q-learning with reward shaping  

![Demo Screenshot](https://drive.google.com/file/d/1K0yN7ca792uxO0zEk8qeOOYJid_jjkTq/view?usp=sharing)

---

## 🔎 Features

- **Procedural Maze Generation**  
  - Random walls (`p=0.1`) with a central “safe zone” to guarantee connectivity  
  - Deterministic via user-supplied seed  
- **Teleoperation Dynamics**  
  - Move, Push, Pull, and collision mechanics in a grid world  
- **Solvability Checker**  
  - Two-phase BFS to filter out unsolvable layouts before play or training  
- **Reinforcement Learning**  
  - Tabular Q-learning (`α=0.1`, `γ=0.95`, ε-greedy)  
  - Dense reward shaping (step penalty, contact bonuses, distance reward, collision/jiggle penalties, goal/failure rewards, baseline adjustment)  
- **Real-Time Visualization**  
  - Pygame UI with modes for Manual, Training, Testing, and Results  
- **Generalization**  
  - High success on held-out mazes (~80–90% on unseen layouts)

---

## 🖥️ Requirements

- Python 3.8+  
- [Pygame](https://www.pygame.org/)  
- Standard libraries: `numpy`, `pickle`

Optional (for experiments & plotting):
- `matplotlib`  
- `pandas`

---

## Installation

```bash
# Clone the repo
git clone https://github.com/harshsenjaliya/Teleoperation-Maze-Game-with-Q-Learning.git
cd Teleoperation-Maze-Game-with-Q-Learning

# (Optional) Create & activate a venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

