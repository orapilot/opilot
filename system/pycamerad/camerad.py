import os
import time
import cv2
import threading

from openpilot.common.realtime import set_realtime_priority
from openpilot.common.basedir import BASEDIR
from openpilot.system.pycamerad.cameras import Cameras

def main():
  set_realtime_priority(5)
  Cameras().starts()

if __name__ == "__main__":
  main()    
