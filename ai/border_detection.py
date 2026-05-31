import cv2
import torch
import datetime
from ultralytics import YOLO


# ── CHECK DEVICE ───────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on : {device}")
print(f"GPU        : {torch.cuda.get_device_name(0) if device == 'cuda' else 'None'}")


# ── LOAD MODEL ─────────────────────────────────────────────────────
model = YOLO("yolov8m.pt")
model.to(device)


# assign an image path and other parameters for the detection
IMAGE_PATH    = "/home/zam/Downloads/thermal image.jpeg"
BORDER_LINE_X = 600
CONFIDENCE    = 0.05
CLASSES       = [0, 2, 5, 7]  #each number presents an objects 


# objects detection boxs colors 
CLASS_COLORS = {
    "person" : (0,   255, 0),    # green if not crossed
    "car"    : (255, 165, 0),    # orange same as above
    
}

CROSSED_COLOR = (0, 0, 255)      # it will draw a red box for any object when crossed


# ── TRACKING STATE ─────────────────────────────────────────────────
crossed_ids   = set()
alert_log     = []


# ── DRAW BORDER LINE ───────────────────────────────────────────────
def draw_border(frame, border_x):
    h, w = frame.shape[:2]
    cv2.line(frame, (border_x, 0), (border_x, h), (0, 0, 255), 3)
    cv2.putText(frame, "BORDER LINE",
                (border_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    return frame


# Create a function to draw box that has all the info about the iimage 
def draw_stats(frame, detections):
    cv2.rectangle(frame, (0, 0), (280, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"Detections : {detections}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Crossings  : {len(alert_log)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Device     : {device.upper()}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    # legend
    cv2.putText(frame, "person=green car=orange bus=purple truck=yellow",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1)
    return frame





# PRINT FINAL REPORT bassed on the detection results
def print_report():
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(f"Device                  : {device.upper()}")
    print(f"Total detections        : {len(alert_log)}")
    print(f"Total crossings         : {len(crossed_ids)}")
    print("-"*50)
    for alert in alert_log:
        print(f"  Time   : {alert['time']}")
        print(f"  Object : {alert['label']} ID:{alert['id']}")
        print(f"  Conf   : {alert['conf']:.0%}")
        print("-"*50)


#  process the image 
##load the image 
def process_image(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: cannot open image — {image_path}")
        return

    height, width = frame.shape[:2]
    print(f"\nImage loaded")
    print(f"Resolution : {width}x{height}")
    print(f"Border line: x={BORDER_LINE_X}")
   

    # create window with sliders to  enhance the detection result
    cv2.namedWindow("Border Detection System")
    cv2.createTrackbar("Border Line", "Border Detection System", BORDER_LINE_X, width,  lambda x: None)
    cv2.createTrackbar("Brightness",  "Border Detection System", 50,            100,    lambda x: None)

    while True:
        # get slider values
        border_line_x = cv2.getTrackbarPos("Border Line", "Border Detection System")
        brightness    = cv2.getTrackbarPos("Brightness",  "Border Detection System")

        # enhance image brightness (for thermal images)
        alpha         = 1.0 + brightness / 50.0
        beta          = brightness
        display_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        #  run detection from YoLO class functionn
        results = model(
            display_frame,
            device=device,
            classes=CLASSES,
            conf=CONFIDENCE,
            iou=0.3,
            verbose=False
        )

        detections = len(results[0].boxes) if results[0].boxes is not None else 0

        # display and calling the draw border and stats functions 
        display_frame = draw_border(display_frame, border_line_x)
        display_frame = draw_stats(display_frame, detections)

        ###process each detection and print info of the detected objects to terminal 
        if results[0].boxes is not None and len(results[0].boxes) > 0:

            boxes   = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs   = results[0].boxes.conf.cpu().numpy()

            for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
                x1, y1, x2, y2 = map(int, box)
                center_x        = (x1 + x2) // 2
                center_y        = (y1 + y2) // 2
                label           = model.names[int(cls)]

                print(f"{label} {idx+1} | Conf: {conf:.0%} | Position: ({center_x}, {center_y})")

                ### logic to determine if an object has crossed the border 
                if center_x < border_line_x: ## if center x is less than border possition it means that the object has crossed the border line 
                    color = CROSSED_COLOR
                    status_text = "CROSSED"
                    if idx not in crossed_ids: ## prevents the same person triggering again if he is already crossed
                     crossed_ids.add(idx)
                     time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                     print(f"ALERT | {time_now} | {label} ID:{idx} crossed the border | conf:{conf:.0%}")
                     alert_log.append({
                        "time"  : time_now,
                        "id"    : idx,
                        "label" : label,
                        "conf"  : conf,
                                        })
                else:
                   color= CLASS_COLORS.get(label, (0, 255, 0))
                   status_text = "SAFE"
                
                ### draw the box  that hilight the detected object 
                cv2.rectangle(display_frame,
                              (x1, y1), (x2, y2), color, 2)

                # ── draw info label on the box
                label_text = f"{label} ID:{idx} {status_text} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label_text,cv2.FONT_HERSHEY_SIMPLEX,0.6, 1)

                cv2.rectangle(display_frame,
                              (x1, y1 - th - 8),
                              (x1 + tw + 4, y1),
                              color, -1)
                
                cv2.putText(display_frame,
                            label_text,
                            (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1)

                #  draw center dot on the object
                cv2.circle(display_frame,
                           (center_x, center_y),
                           6, color, -1)

        # Display the detection window 
        cv2.imshow("Display my Drone Border Detection Window ", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print_report()


 
process_image(IMAGE_PATH)