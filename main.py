import win32api
import win32gui
import win32ui
import win32con
from PIL import ImageGrab
import PIL
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from digitTrain import MyModel
from digitTrain import MyDataSet
import sys
from multiprocessing import Process, Queue
sys.path.append('yolov5-master')
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from multiprocessing import Process, Queue



def DetectProcess(q):
    print("proc start")
    width = 7
    height = 11
    yLoc = 557
    xLocs = (877,884,895,902)
    
    
    digitModel = MyModel.load_from_checkpoint("digit.ckpt")
    digitModel.cuda()
    digitModel.eval()
    
    device = select_device('0')
    model = DetectMultiBackend('best.pt', device=device, dnn=False, data='')
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size((640,640), s=stride)
    model.model.float()
    model.warmup(imgsz=(1, 3, *imgsz), half=False)
    
    while True:
        item = {}
        item["rects"] = []
        item["leads"] = []

        img_ready = ImageGrab.grab((0, 0, 1920, 1080))
        img_arr = np.array(img_ready)
        digits = np.zeros((0,height,width,3))
        for x in xLocs:
            digit = img_arr[yLoc:yLoc+height, x:x+width, :].reshape(1,height,width,3)
            digits = np.concatenate((digits,digit),axis = 0)
    
        torch_digits = torch.Tensor((digits/255-0.5)*2)
        
        digit_result = torch.argmax(digitModel.model(torch_digits.cuda()),dim = 1)
        
        num_result = np.array(digit_result.cpu())
        txt = ""
        
        for i in range(len(num_result)):
            num = num_result[i]
            if num == 10:
                num = 0
            
            txt += str(num)
            if(i==1):
                txt += "."
        y = yLoc+height+2
        x = xLocs[0]
        
        seconds = float(txt)
        lead = seconds/3.6
        item["textArgs"] = txt,5,(x,y,x+width*5,y+height),win32con.DT_TOP
                
        y = y+height+2
        item["leadArgs"] = str(lead),3,(x+width,y,x+width*4,y+height),win32con.DT_TOP
        
        screen_arr = np.array(img_ready.resize((640,640),PIL.Image.BICUBIC)).transpose((2,0,1))
        screen_arr = np.ascontiguousarray(screen_arr)/255.0
        screen_arr = screen_arr[None]
        pred = model(torch.from_numpy(screen_arr).cuda().float(), augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=50)
        
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords((640,640), det[:, :4], (1080,1920)).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
        
                # Write results
                for *xyxy, conf, cls in reversed(det):               
                    xyxy[0] = (xyxy[0]/640*1920).round()
                    xyxy[1] = (xyxy[1]/640*1080).round()
                    xyxy[2] = (xyxy[2]/640*1920).round()
                    xyxy[3] = (xyxy[3]/640*1080).round()
                    
                    item["rects"] += [[int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]]
                    
                    length = float(xyxy[2]-xyxy[0])*lead
                    h = (xyxy[3]-xyxy[1])*2
                    
                    
                    if cls < 0.5:
                        realLead = (xyxy[2]+length).round()
                        item["leads"] += [[int(realLead),int(h),int(xyxy[2]),int(xyxy[3])]]
                    else:
                        realLead = (xyxy[0]-length).round()
                        item["leads"] += [[int(realLead),int(h),int(xyxy[2]),int(xyxy[3])]]
        if(q.empty()):
            q.put(item)
        
        
def myDrawRectangle(hdc, x1, y1, x2, y2):
    win32gui.Polyline(hdc, ((x1,y1),(x1,y2),(x2,y2),(x2,y1),(x1,y1)))

def myDrawLead(hdc,realLead,height,x,y):
    win32gui.Polyline(hdc, ((x,y),(realLead,y)))
    win32gui.Polyline(hdc, ((realLead,y),(realLead,y+height)))


if __name__ == "__main__":
    hwnd = win32gui.FindWindow(None, "《戰艦世界》")
    hdc = win32gui.GetDC(hwnd)
    #hdc = win32gui.GetDC(0)
    hpen = win32gui.CreatePen(win32con.PS_GEOMETRIC,2, win32api.RGB(255,0,0))
    win32gui.SelectObject(hdc,hpen)
    win32gui.SetTextColor(hdc, win32api.RGB(255,255,255))
    win32gui.SetBkColor(hdc,win32api.RGB(0,0,0))
    font = win32ui.CreateFont({'height':11,'width':7})
    win32gui.SelectObject(hdc,font.GetSafeHandle())
    
    
    q = Queue()
    item = None
    procDectect = Process(target=DetectProcess,args=(q,))
    procDectect.start()
    while True:
        if not q.empty():
            item = q.get(True)
        if item:
            win32gui.DrawText(hdc, *item["textArgs"])
            win32gui.DrawText(hdc, *item["leadArgs"])
            for i in range(len(item["rects"])):
                myDrawRectangle(hdc,*(item["rects"][i]))
                myDrawLead(hdc,*(item["leads"][i]))
    