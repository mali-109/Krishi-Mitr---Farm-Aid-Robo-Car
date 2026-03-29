# Flask API for KrishiMitr Robot

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import os
import time

import RPi.GPIO as GPIO
from gpiozero import AngularServo

import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_dht

import cv2
from PIL import Image
import torch
from torchvision import transforms, models

# -------- Flask --------
app = Flask(__name__)
CORS(app)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
IMAGE_PATH = os.path.join(STATIC_DIR, "latest.jpg")

# -------- GPIO PINS --------
IN1, IN2, IN3, IN4 = 17, 27, 22, 23
SERVO_PIN = 18
SPRAY_PIN = 20
GAS_PIN = 12
TRIG = 24
ECHO = 25

motor_pins = [IN1, IN2, IN3, IN4]

# -------- GPIO Setup --------
def setup_gpio():

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    for p in motor_pins:
        GPIO.setup(p, GPIO.OUT)
        GPIO.output(p, GPIO.LOW)

    GPIO.setup(SPRAY_PIN, GPIO.OUT)
    GPIO.output(SPRAY_PIN, GPIO.LOW)

    GPIO.setup(GAS_PIN, GPIO.IN)

    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

# -------- Servo --------
servo = AngularServo(SERVO_PIN, min_angle=0, max_angle=180)

def insert_sensor():
    servo.angle = 90
    time.sleep(2)

def remove_sensor():
    servo.angle = 0
    time.sleep(1)

# -------- Ultrasonic --------
def get_distance():

    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start = time.time()
    stop = time.time()

    timeout = start + 0.04

    while GPIO.input(ECHO) == 0 and time.time() < timeout:
        start = time.time()

    while GPIO.input(ECHO) == 1 and time.time() < timeout:
        stop = time.time()

    duration = stop - start
    distance = duration * 17150
    distance = round(distance, 2)

    # Filter bad readings
    if distance <= 2 or distance >= 400:
        return 999

    return distance

# -------- Motor Control --------
def motor_stop():

    for p in motor_pins:
        GPIO.output(p, GPIO.LOW)

def move_car(direction):

    motor_stop()

    if direction == "forward":

        distance = get_distance()
        print("Distance:", distance)

        if 5 < distance <= 10:
            print("Obstacle detected - stopping")
            return

        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    elif direction == "backward":

        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    elif direction == "left":

        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    elif direction == "right":

        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

# -------- Sensors Setup --------
i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS.ADS1115(i2c)
soil_channel = AnalogIn(ads, 0)

dht_device = adafruit_dht.DHT22(board.D4)

def read_soil_percent():

    insert_sensor()

    voltage = soil_channel.voltage
    percent = (1 - (voltage / 3.3)) * 100

    remove_sensor()

    return round(percent, 2)

def read_dht():

    for _ in range(3):
        try:
            t = dht_device.temperature
            h = dht_device.humidity

            if t is not None and h is not None:
                return round(t, 2), round(h, 2)

        except Exception:
            time.sleep(0.5)

    return None, None

def read_gas():

    value = GPIO.input(GAS_PIN)

    if value == 0:
        return "Gas Detected"
    else:
        return "Air Normal"

# -------- Spray --------
def spray_on():
    GPIO.output(SPRAY_PIN, GPIO.HIGH)

def spray_off():
    GPIO.output(SPRAY_PIN, GPIO.LOW)

# -------- AI Model --------
device = torch.device("cpu")

MODEL_PATH = "/home/satyam/plant_disease_model.pth"

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

classes = ['Alternaria', 'Leaf_blight', 'Healthy']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -------- Camera --------
def capture_frame():

    cap = cv2.VideoCapture(0)
    time.sleep(0.5)

    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(IMAGE_PATH, frame)
        return True

    return False

def detect_on_image():

    img = Image.open(IMAGE_PATH).convert("RGB")

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)
        prob = torch.softmax(output, dim=1)

        conf, pred = torch.max(prob, 1)

    label = classes[pred.item()]

    return label, float(conf.item())

# -------- Routes --------

@app.route("/")
def home():

    return "🌾 KrishiMitr Flask Server Running"

@app.route("/control")
def control():

    cmd = request.args.get("cmd")

    if cmd == "stop":
        motor_stop()
    else:
        move_car(cmd)

    return jsonify({"cmd": cmd})

@app.route("/sensor")
def sensor():

    soil = read_soil_percent()
    temp, hum = read_dht()
    gas = read_gas()
    distance = get_distance()

    return jsonify({
        "moisture": soil,
        "temperature": temp,
        "humidity": hum,
        "gas": gas,
        "distance": distance
    })

@app.route("/spray_on")
def spray_start():

    spray_on()
    return jsonify({"spray": "on"})

@app.route("/spray_off")
def spray_stop():

    spray_off()
    return jsonify({"spray": "off"})

@app.route("/capture")
def capture_and_detect():

    if not capture_frame():
        return jsonify({"error": "camera capture failed"})

    label, confidence = detect_on_image()

    return jsonify({
        "label": label,
        "confidence": confidence
    })

@app.route("/view")
def view_image():

    response = make_response(send_file(IMAGE_PATH, mimetype='image/jpeg'))
    response.headers['Cache-Control'] = 'no-cache'

    return response

# -------- Start Server --------
if __name__ == "__main__":

    setup_gpio()

    app.run(host="0.0.0.0", port=5000)
