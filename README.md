# spatial-camera

## How to install:
##### 1. Install Python 3.11 (one-time)
`brew install python@3.11`

##### 2. Make / activate a fresh venv using that interpreter
`/opt/homebrew/bin/python3.11 -m venv ~/eye-hand-env`   # path may be /usr/local/... on Intel
`source ~/eye-hand-env/bin/activate`

##### 3. Upgrade pip and install deps
`python -m pip install --upgrade pip`
`pip install opencv-python mediapipe numpy pyautogui`

##### 4. Run it
`python eye_hand_tracking.py`
press "p" or "pinch" to validate dot
