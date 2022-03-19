from ctypes import *
import PyHook3 as pyHook
import pythoncom
import time
from os import walk
from PIL import ImageGrab

plot_interval = 10

record = 0
stop = 0
prev_KeyID = 0
def onKeyboardEvent(event):
    global prev_KeyID
    if event.Alt > 0 and chr(event.KeyID) == 'G' and prev_KeyID != event.KeyID:
        global record
        record ^= 1
    if event.Alt > 0 and chr(event.KeyID) == 'S' and prev_KeyID != event.KeyID:
        global stop
        stop ^= 1
    
    prev_KeyID = event.KeyID
    return True

last_time = -1
double_click = False
def OnMouseEvent(event):
  global last_time
  global double_click
  cur_time = event.Time
  if cur_time - last_time < 200:
      double_click = True
      last_time = -1
  else:
      last_time = cur_time
  return True

hm = pyHook.HookManager()
hm.KeyDown = onKeyboardEvent
hm.MouseLeftUp = OnMouseEvent
hm.HookMouse()
hm.HookKeyboard()

i=0




prev_record = -1
prev_time = time.time()
while True:
    if stop:
        break
        
    pythoncom.PumpWaitingMessages()
    if prev_record != record:
        if record:
            print('start recording')
        else:
            print('stop recording')
        prev_record = record
    
    if record and double_click:
        img = ImageGrab.grab((0, 0, 1920, 1080))
        img.save('./SmokeScreenShots/'+str(time.time())+'.bmp',format='bmp')
        print('saving image: ', time.time())
        prev_time = time.time()
        double_click = False
        
    time.sleep(0.01)
    i+=1

hm.UnhookMouse() 
hm.UnhookKeyboard()
