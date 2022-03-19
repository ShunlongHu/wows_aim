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

hm = pyHook.HookManager()
i=0




prev_record = -1
prev_time = time.time()
while True:
    if stop:
        break
    if time.time()-prev_time >= plot_interval and record:
        img = ImageGrab.grab((0, 0, 1920, 1080))
        img.save('./SmokeScreenShots/'+str(time.time())+'.bmp',format='bmp')
        print('saving image: ', time.time())
        prev_time = time.time()
    pythoncom.PumpWaitingMessages()
    if prev_record != record:
        if record:
            print('start recording')
        else:
            print('stop recording')
        prev_record = record
    
        
    time.sleep(0.01)
    i+=1

hm.UnhookKeyboard()
