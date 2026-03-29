from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import datetime
import os
from openai import OpenAI   # ✅ NEW

app = Flask(__name__)

DATA_FILE = "data/sensor_data.csv"

os.makedirs("data", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("static/pdfs", exist_ok=True)

# ✅ NEW GROQ (OpenAI STYLE)
client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

PI_IP = None


@app.route("/", methods=["GET", "POST"])
def dashboard():

    global PI_IP

    if request.method == "POST":
        PI_IP = request.form.get("pi_ip")

    # -------- LOAD DATA --------
    try:
        df = pd.read_csv(DATA_FILE)
    except:
        df = pd.DataFrame(columns=[
            "time","temperature","humidity","soil","gas","distance","health"
        ])

    # -------- FIX MISSING COLUMNS --------
    required_cols = ["time","temperature","humidity","soil","gas","distance","health"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # -------- SENSOR DATA --------
    if PI_IP:
        try:
            res = requests.get(f"http://{PI_IP}:5000/sensor", timeout=5)
            data = res.json()

            new_row = {
                "time": datetime.datetime.now(),
                "temperature": data.get("temperature"),
                "humidity": data.get("humidity"),
                "soil": data.get("moisture"),
                "gas": data.get("gas"),
                "distance": data.get("distance"),
                "health": 100
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)

        except:
            print("Could not connect to Raspberry Pi")

    # -------- GRAPHS --------
    if len(df) > 0:

        def safe_plot(column, title, file):
            try:
                plt.figure()
                plt.plot(df[column])
                plt.title(title)
                plt.xlabel("Reading")
                plt.savefig(f"static/{file}")
                plt.close()
            except:
                pass

        safe_plot("temperature", "Temperature", "temp_graph.png")
        safe_plot("humidity", "Humidity", "humidity_graph.png")
        safe_plot("soil", "Soil Moisture", "soil_graph.png")
        safe_plot("distance", "Distance", "distance_graph.png")
        safe_plot("health", "Plant Health", "health_graph.png")

    # -------- GAS --------
    gas_alert = False
    gas_status = "Unknown"

    if len(df) > 0:
        latest = df.iloc[-1]
        gas_status = latest.get("gas", "Unknown")

        if gas_status == "Gas Detected":
            gas_alert = True

    table = df.tail(10).to_html(index=False)

    # -------- PDF LOAD --------
    pdf_files = [f for f in os.listdir("static/pdfs") if f.endswith(".pdf")]
    print("PDF FILES:", pdf_files)

    return render_template(
        "index.html",
        table=table,
        gas_alert=gas_alert,
        gas_status=gas_status,
        pdfs=pdf_files
    )


# -------- SERVE PDF --------
@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    return send_from_directory("static/pdfs", filename)


# -------- CLEAR DATA --------
@app.route("/clear", methods=["POST"])
def clear_data():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    return jsonify({"status": "data_cleared"})


# -------- CAPTURE --------
@app.route("/capture", methods=["POST"])
def capture():

    global PI_IP

    if not PI_IP:
        return jsonify({"status": "pi_not_connected"})

    try:
        res = requests.get(f"http://{PI_IP}:5000/capture", timeout=5)
        return jsonify(res.json())
    except:
        return jsonify({"status": "error"})


# -------- UPDATED CHATBOT --------
@app.route("/chat", methods=["POST"])
def chat():

    try:
        data = request.json
        message = data.get("message")
        language = data.get("language", "English")

        # -------- SENSOR DATA --------
        try:
            df = pd.read_csv(DATA_FILE)
            latest = df.iloc[-1]

            sensor_info = f"""
Temperature: {latest['temperature']} °C
Humidity: {latest['humidity']} %
Soil Moisture: {latest['soil']} %
Gas: {latest['gas']}
Distance: {latest['distance']} cm
Plant Health: {latest['health']}
"""
        except:
            sensor_info = "No sensor data available."

        prompt = f"""
You are an agriculture expert for Indian farmers.

Farm Data:
{sensor_info}

Give simple and practical advice.

Answer in {language}.

Question:
{message}
"""

        # ✅ NEW WORKING API CALL
        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=prompt
        )

        reply = response.output_text

        return jsonify({"reply": reply})

    except Exception as e:
        print(e)
        return jsonify({"reply": "AI assistant unavailable"})


if __name__ == "__main__":
    app.run(debug=True)