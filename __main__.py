import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image

# Configure Google Generative AI
genai.configure(api_key="AIzaSyBfm26zkH8why393tQDFPesXg6hmiIFu8Y")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Adjust camera index if needed (0, 1, 2, etc.)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas, eraser_mode):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = tuple(lmList[8][0:2])
        if prev_pos is None:
            prev_pos = current_pos
        if eraser_mode:
            cv2.line(canvas, current_pos, prev_pos, (0, 0, 0), 50)  # Eraser with thicker line
        else:
            cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    return current_pos, canvas

def sendToAI(model, canvas):
    pil_image = Image.fromarray(canvas)
    response = model.generate_content(["Solve this math problem", pil_image])
    return response.text

def drawText(img, text, position, font, scale, color, thickness, max_width):
    words = text.split(' ')
    line = ''
    y = position[1]

    for word in words:
        if cv2.getTextSize(line + word, font, scale, thickness)[0][0] < max_width:
            line += word + ' '
        else:
            cv2.putText(img, line, (position[0], y), font, scale, color, thickness, lineType=cv2.LINE_AA)
            line = word + ' '
            y += int(scale * 30)  # Move to the next line

    cv2.putText(img, line, (position[0], y), font, scale, color, thickness, lineType=cv2.LINE_AA)

prev_pos = None
canvas = None
output_text = ""
output_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
clear_canvas = False
eraser_mode = False

# Main loop
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        if fingers == [1, 0, 0, 0, 1]:
            eraser_mode = not eraser_mode  # Toggle eraser mode
        prev_pos, canvas = draw(info, prev_pos, canvas, eraser_mode)

        if fingers == [1, 1, 1, 1, 1]:
            output_text = sendToAI(model, canvas)
            print(f"Fingers: {fingers}, Landmarks: {lmList}")
            clear_canvas = True

    if clear_canvas:
        output_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        drawText(output_canvas, output_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, max_width=1180)
        print(output_text)

        canvas = np.zeros_like(img)
        prev_pos = None
        clear_canvas = False

    cv2.imshow("Canvas", canvas)
    cv2.imshow("Output", output_canvas)
    cv2.imshow("Combined", cv2.addWeighted(img, 0.7, canvas, 0.3, 0))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
