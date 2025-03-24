# GesturaMath: AI-Powered Mathematical Gestures
This project involves real-time hand gesture recognition using OpenCV and the cvzone library for hand tracking. It also integrates Google Generative AI for interacting with the detected gestures. The program uses a webcam to capture the video feed, detects hand gestures, and performs actions such as drawing on a canvas or clearing the canvas based on the recognized gestures. Additionally, it sends the canvas image to the Google Generative AI model to solve a math problem drawn on the canvas.

## Requirements
Python 3.x
* OpenCV (opencv-python and opencv-python-headless)
* cvzone (cvzone)
* NumPy (numpy)
* google-generativeai (google-generativeai)
## Installation
Install the required Python packages:

```bash
pip install opencv-python opencv-python-headless cvzone numpy pillow google-generativeai
```
Configure Google Generative AI by setting up your API key:

```bash
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
```
## Usage
Configure API Key: Set up your Google Generative AI API key in the script:

```bash
genai.configure(api_key="YOUR_API_KEY")
```
Run the Script: Execute the script to start the webcam and begin hand gesture recognition:

```bash
hand_gesture_recognition.py
```
## Gesture Commands:
* Index Finger Up: Draw on the canvas.
* Thumb Up: Clear the canvas.
* Four Fingers Up: Send the canvas image to the AI for solving a math problem.
##  Explanation of the Script
Importing Libraries
```bash
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
```
* cv2: OpenCV library for computer vision tasks.
* cvzone.HandTrackingModule: For hand tracking and gesture recognition.
* numpy: For numerical operations.
* google.generativeai: For interacting with Google Generative AI.
* PIL: For image processing.
## Configuring Google Generative AI
```bash
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')
```
* Configure the Google Generative AI with the provided API key.
## Initializing Webcam and HandDetector
```bash
cap = cv2.VideoCapture(0)  # Adjust camera index if needed (0, 1, 2, etc.)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)
```
* Initialize the webcam and set the resolution.
* Initialize the HandDetector with appropriate settings.
## Hand Gesture Detection
```bash
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None
```
* Detect hands in the image and return the list of fingers and landmarks.
## Drawing on the Canvas
```bash
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = tuple(lmList[8][0:2])
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(canvas)
    return current_pos, canvas
```
* Draw on the canvas if the index finger is up.
* Clear the canvas if the thumb is up.
## Sending Canvas to Google Generative AI
```bash
def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""
```
* Send the canvas image to Google Generative AI to solve a math problem if four fingers are up.
## Main Loop
```bash
prev_pos = None
canvas = None
output_text = ""

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
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)
        print(f"Fingers: {fingers}, Landmarks: {lmList}")

    if output_text:
        print(output_text)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("image_combined", cv2.addWeighted(img, 0.7, canvas, 0.3, 0))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
* Capture frames from the webcam and process them.
* Display the original image, canvas, and combined image.
* Exit the loop when 'q' is pressed.

By following these instructions, you can set up and run the real-time hand gesture recognition program with Google Generative AI integration.
