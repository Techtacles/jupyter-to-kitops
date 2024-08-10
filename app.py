import joblib
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
model = joblib.load("models/scikit_class_model_v2.joblib")
scaler = joblib.load("models/scaler")
feature_gen = joblib.load("models/feature_gen")
app = Flask(__name__)


def predict_function(Gender, Customer, Age, Travel_Type, Class,
                     FlightDistance, WifiService, DepartureArrivalTime,
                     BookingEase, GateLocation, FoodAndDrinks, Boarding,
                     SeatComfort, Entertainment, OnboardService,
                     LegRoomService, Baggage, CheckinService, InflightService, 
                     Cleanliness, DepartureDelay, ArrivalDelay):

    data = pd.DataFrame([[Gender, Customer, Age,
                                   Travel_Type, Class, FlightDistance,
                                   WifiService, DepartureArrivalTime,
                                   BookingEase, GateLocation, FoodAndDrinks,
                                   Boarding, SeatComfort, Entertainment,
                                   OnboardService, LegRoomService, Baggage,
                                   CheckinService, InflightService,
                                   Cleanliness, DepartureDelay,
                                   ArrivalDelay]],
                        columns=['Gender', 'Customer Type', 'Age',
                                 'Type of Travel', 'Class',
                                 'Flight Distance', 'Inflight wifi service',
                                 'Departure/Arrival time convenient',
                                 'Ease of Online booking', 'Gate location',
                                 'Food and drink', 'Online boarding',
                                 'Seat comfort',
                                 'Inflight entertainment', 'On-board service',
                                 'Leg room service', 'Baggage handling',
                                 'Checkin service', 'Inflight service',
                                 'Cleanliness', 'Departure Delay in Minutes',
                                 'Arrival Delay in Minutes'])
    input_transformed = feature_gen.transform(data)
    print(data)
    input_normalized = scaler.transform(input_transformed)
    prediction = model.predict(input_normalized)
    return prediction[0]

# print(predict_function("Male","Loyal" "Customer",13,"Personal Travel","Eco Plus",460,3,4,3,1,5,3,5,5,4,3,4,4,5,5,25,18.0))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form["gender"]
    customer = request.form["customer"]
    age = int(request.form["age"])
    travel_type = request.form["travel_type"]
    flight_class = request.form["flight_class"]
    flight_distance = int(request.form["flight_distance"])
    wifi_service = int(request.form["wifi_service"])
    time = int(request.form["time"])
    booking = int(request.form["booking"])
    gate_location = int(request.form["gate_location"])
    food_and_drinks = int(request.form["food_drinks"])
    boarding = int(request.form["boarding"])
    seat_comfort = int(request.form["seat_comfort"])
    entertainment = int(request.form["entertainment"])
    onboard_service = int(request.form["onboard_service"])
    leg_room_service = int(request.form["leg_room_service"])
    baggage = int(request.form["baggage"])
    checkinservice = int(request.form["checkinservice"])
    inflight_service = int(request.form["inflight_service"])
    cleanliness = int(request.form["cleanliness"])
    departure_delay = int(request.form["departure_delay"])
    arrival_delay = int(request.form["arrival_delay"])
    prediction = predict_function(gender, customer, age,
                                  travel_type, flight_class, flight_distance,
                                  wifi_service, time, booking, gate_location,
                                  food_and_drinks, boarding, seat_comfort,
                                  entertainment, onboard_service,
                                  leg_room_service, baggage, checkinservice,
                                  inflight_service, cleanliness,
                                  departure_delay, arrival_delay)
    if prediction == 0:
        prediction = "neutral or dissatisfied"
    else:
        prediction = "satisfacted"

    return render_template("index.html",
                           prediction_text=f"The  prediction is {prediction}")

if __name__ == '__main__':  
    app.run()
