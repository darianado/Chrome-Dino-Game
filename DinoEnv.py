from mss import mss
#import pydirectinput
import pyautogui
import cv2
import numpy as np
import pytesseract
import time
#from gym import Env
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

class DinoGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape = (1,83,100), dtype = np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {'top':300, 'left':0, 'width':400, 'height':250}
        self.done_location = {'top':330, 'left':240, 'width':240, 'height':35}
        
                
        
    def step(self, action):
        # Actions 0 = Space, 1 = Down, 2 = No action
        action_map = {0: 'space',
                     1 : 'down',
                     2 : 'no_op'}

        if action != 2:
            pyautogui.keyDown(action_map[action])
            time.sleep(0.1)
            pyautogui.keyUp(action_map[action])
            
        # Check if the game is done
        done, done_cap = self.get_done()
        
        # Get new obs
        new_obs = self.get_observation()
        
        # Reward
        reward = -100 if done else (3 if action == 2 else 1)
        
        # info dict - that's what stablebaselines3 expects that's why it's defined
        info = {}
        
        return new_obs, reward, done, False, info
        
    def render(self):
        obs = np.array(self.cap.grab(self.game_location))[:,:,:3]
        #plt.imshow(obs)
        cv2.imshow('Game', obs)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            self.close()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        time.sleep(1)
        pyautogui.click(x = 150, y = 150)
        pyautogui.keyDown('space')
        time.sleep(0.1)
        pyautogui.keyUp('space')
        info = {}
        return self.get_observation(), info
        
        
    def get_observation(self):
        # Take the screen capture
        # It returns 4 channels so slicing just first 3 channels
        raw_obs = np.array(self.cap.grab(self.game_location))[:,:,:3]
        # Grayscale observation
        gray_obs = cv2.cvtColor(raw_obs, cv2.COLOR_BGR2GRAY)
        # Resize the image
        resized_obs = cv2.resize(gray_obs,(100,83))
        adjusted_image = cv2.convertScaleAbs(resized_obs, alpha=1.5, beta=0)
        # See the channels first - because stablebaselines3 requires it!
        channel = np.reshape(adjusted_image,(1,83,100))
        
        return channel
        
    def get_done(self):
        # Get 'Game Over' text on the screen
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        # Valid done text
        done_strings = ['GAME', 'GAHE']

        # Set done initially to false
        done = False
        # Convert image to string using pytesseract
        res = pytesseract.image_to_string(done_cap)[:4]
        # Check if the string we converted in the done_strings
        if res in done_strings:
            # Return true if it captures 'GAME OVER'
            done = True
        
        return done, done_cap
        
    def close(self):
        # close all windows which opened while rendering
        cv2.destroyAllWindows()