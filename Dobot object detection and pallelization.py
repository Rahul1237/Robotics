import cv2
import time
from ultralytics import YOLO
from pydobot.dobot import MODE_PTP
import pydobot

# ================== Robot Setup ==================
device = pydobot.Dobot(port="/dev/ttyACM0")
device.speed(10, 10)
device.home()

# ================== Camera Setup ==================
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================== YOLO Setup ==================
model = YOLO("yolov8s.pt")

# ================== Robot Coordinates (adjust!) ==================
home_coordinates = [239.999, 0.0, 150.0, -8.881]
intermediate_coordinates = [193.526, 22.005, 35.189, 6.487]
pickup_coordinates = [306.420, -82.706, 60.320, -15.104]
palleteA_coordinates = [210.180, -233.553, 24.075, -46.766]
palleteB_coordinates = [296.191, 193.261, 25.764, 33.123]


def is_home():
    """Check if robot is at home position"""
    pose = device.get_pose()
    try:
        current_pose = (pose.position.x, pose.position.y, pose.position.z, pose.position.r)
    except AttributeError:
        try:
            current_pose = (pose.x, pose.y, pose.z, pose.r)
        except AttributeError:
            current_pose = tuple(pose[:4]) if hasattr(pose, '_iter_') else pose

    home_pose = tuple(home_coordinates)
    position_tol = 10  # mm tolerance
    is_at_home = all(abs(current_pose[i] - home_pose[i]) < position_tol for i in range(3))
    return is_at_home


# ================== Main Loop ==================
try:
    print("Starting object detection and palletization...")
    print("Press 'q' to quit")

    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

    detection_enabled = True  # Flag to pause detection during robot moves

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error, skipping frame...")
            continue

        annotated_frame = frame.copy()
        detected_class, detected_label = None, "None"

        # ============ Run YOLO Only When Enabled ============
        if detection_enabled and is_home():
            results = model(frame, verbose=False)
            r = results[0]
            annotated_frame = r.plot()

            if r.boxes is not None and len(r.boxes) > 0:
                names = r.names
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = names[cls_id].lower()

                    print(f"Detected: {label} (confidence: {conf:.2f})")

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.putText(annotated_frame,
                                f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if conf > 0.75:
                        if label in ["car", "bicycle", "airplane"]:
                            detected_class = "vehicle"
                            detected_label = f"{label} ({conf:.2f})"
                            break
                        elif label in ["apple", "banana", "pizza"]:
                            detected_class = "food"
                            detected_label = f"{label} ({conf:.2f})"
                            break

        # Display status on feed
        status_text = f"Status: {detected_class if detected_class else 'Waiting...'}"
        label_text = f"Object: {detected_label}"
        cv2.putText(annotated_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, label_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Camera Feed', annotated_frame)

        # Quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed")
            break

        # ============ Robot Actions ============
        if detected_class == "food":
            print("Food detected → moving to Pallet A")
            detection_enabled = False

            device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=pickup_coordinates[0],
                           y=pickup_coordinates[1], z=pickup_coordinates[2], r=pickup_coordinates[3])
            time.sleep(0.3)
            device.suck(True)
            time.sleep(0.5)

            device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=intermediate_coordinates[0],
                           y=intermediate_coordinates[1], z=intermediate_coordinates[2], r=intermediate_coordinates[3])

            device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=palleteA_coordinates[0],
                           y=palleteA_coordinates[1], z=palleteA_coordinates[2], r=palleteA_coordinates[3])
            time.sleep(0.3)
            device.suck(False)
            time.sleep(0.5)

            device.home()
            time.sleep(2)

            detection_enabled = True

        elif detected_class == "vehicle":
            print("Vehicle detected → moving to Pallet B")
            detection_enabled = False

            device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=pickup_coordinates[0],
                           y=pickup_coordinates[1], z=pickup_coordinates[2], r=pickup_coordinates[3])
            time.sleep(0.3)
            device.suck(True)
            time.sleep(0.5)

            device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=intermediate_coordinates[0],
                           y=intermediate_coordinates[1], z=intermediate_coordinates[2], r=intermediate_coordinates[3])

            device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=palleteB_coordinates[0],
                           y=palleteB_coordinates[1], z=palleteB_coordinates[2], r=palleteB_coordinates[3])
            time.sleep(0.3)
            device.suck(False)
            time.sleep(0.5)

            device.home()
            time.sleep(2)

            detection_enabled = True

        else:
            time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping program...")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()
    if cap is not None:
        cap.release()
    device.close()
    print("Cleanup complete")