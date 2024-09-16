import streamlit as st
import requests
from os import getenv
from datetime import datetime

API_URL = getenv("API_URL")

make_options = ['volkswagen', 'Mitsubishi', 'Audi', 'ford', 'Geo', 'Hyundai', 'jeep', 'FIAT', 'Acura', 'dodge', 'hyundai', 'oldsmobile', 'acura', 'hyundai tk', 'Porsche', 'gmc truck', 'subaru', 'mercury', 'Mercury', 'Ram', 'MINI', 'Plymouth', 'cadillac', 'dodge tk', 'Saab', 'Maserati', 'maserati', 'porsche', 'Lamborghini', 'Dodge', 'Jeep', 'airstream', 'mitsubishi', 'ford truck', 'Saturn', 'buick', 'Mazda', 'Oldsmobile', 'suzuki', 'chevrolet', 'mercedes', 'chev truck', 'audi', 'Volvo', 'HUMMER', 'gmc', 'Lexus', 'BMW', 'Tesla', 'Kia', 'Rolls-Royce', 'Lincoln', 'Daewoo', 'Land Rover', 'Honda', 'lincoln', 'Fisker', 'nissan', 'Ferrari', 'Suzuki', 'Infiniti', 'plymouth', 'Bentley', 'Ford', 'Pontiac', 'Cadillac', 'Mercedes-Benz', 'landrover', 'GMC', 'Volkswagen', 'pontiac', 'Nissan', 'kia', 'toyota', 'land rover', 'vw', 'honda', 'chrysler', 'mazda', 'Jaguar', 'mercedes-b', 'bmw', 'Isuzu', 'Aston Martin', 'Subaru', 'Toyota', 'Buick', 'dot', 'Chrysler', 'smart', 'Scion', 'lexus', 'Chevrolet', 'Other']
body_options = ['G Coupe', 'Beetle Convertible', 'g coupe', 'Van', 'wagon', 'G37 Coupe', 'beetle convertible', 'Koup', 'elantra coupe', 'Crew Cab', 'G Sedan', 'hatchback', 'mega cab', 'q60 coupe', 'q60 convertible', 'Hatchback', 'g convertible', 'promaster cargo van', 'Transit Van', 'Wagon', 'supercab', 'Cab Plus', 'Minivan', 'Q60 Coupe', 'Regular Cab', 'GranTurismo Convertible', 'cts-v coupe', 'van', 'Access Cab', 'Navitgation', 'Club Cab', 'SUV', 'koup', 'access cab', 'Xtracab', 'CTS Coupe', 'g37 coupe', 'Elantra Coupe', 'G Convertible', 'sedan', 'coupe', 'granturismo convertible', 'convertible', 'club cab', 'CrewMax Cab', 'Ram Van', 'TSX Sport Wagon', 'Cab Plus 4', 'CTS-V Coupe', 'extended cab', 'E-Series Van', 'genesis coupe', 'SuperCab', 'king cab', 'crew cab', 'Coupe', 'supercrew', 'double cab', 'tsx sport wagon', 'minivan', 'Q60 Convertible', 'crewmax cab', 'Promaster Cargo Van', 'quad cab', 'CTS-V Wagon', 'King Cab', 'Genesis Coupe', 'CTS Wagon', 'Convertible', 'regular cab', 'g sedan', 'SuperCrew', 'regular-cab', 'suv', 'cab plus 4', 'xtracab', 'Double Cab', 'cts coupe', 'Mega Cab', 'Quad Cab', 'transit van', 'g37 convertible', 'G37 Convertible', 'Sedan', 'Extended Cab', 'e-series van', 'Other']
transmission_options = ['sedan', 'automatic', 'Sedan', 'manual', 'Other']
state_options = ['wi', 'mi', 'ne', 'ga', 'nv', 'va', 'qc', 'oh', 'tn', 'or', 'pr', 'in', 'hi', 'wa', 'az', 'co', 'ut', 'md', 'ny', 'ms', 'nj', 'il', 'ns', 'ok', 'al', 'sc', 'pa', 'on', 'tx', 'fl', 'ma', 'ab', 'mo', 'mn', 'nm', 'nc', 'ca', 'la']
color_options = ['purple', 'pink', 'black', '—', 'blue', 'orange', 'beige', 'white', 'lime', 'charcoal', 'yellow', 'silver', '12655', '18384', 'green', '9562', 'burgundy', 'turquoise', '2172', 'red', 'brown', 'gray', 'off-white', 'gold']
interior_options = ['purple', 'burgundy', 'no', 'white', 'orange', 'beige', 'black', 'brown', '—', 'red', 'silver', 'yellow', 'gray', 'green', 'blue', 'tan', 'off-white', 'gold']

def main():
    st.title("Car Price Prediction")

    year = st.number_input("The manufacturing year of the vehicle.", min_value=1900, max_value=2100, value=2020)
    make = st.selectbox("The brand or manufacturer of the vehicle.", make_options)
    body = st.selectbox("The body type of the vehicle", body_options)
    transmission = st.selectbox("The type of transmission in the vehicle", transmission_options)
    state = st.selectbox("The state where the vehicle is registered.", state_options)
    condition = st.number_input("Condition of the vehicle, possibly rated on a scale.", min_value=0, max_value=100, value=50)
    odometer = st.number_input("The mileage or distance traveled by the vehicle.", min_value=0.0, value=50000.0)
    color = st.selectbox("Exterior color of the vehicle.", color_options)
    interior = st.selectbox("Interior color of the vehicle.", interior_options)
    mmr = st.number_input("Manheim Market Report, possibly indicating the estimated market value of the vehicle.", min_value=0.0, value=15000.0)
    sale_date = st.date_input("The date and time when the vehicle was sold.", value=datetime(2024, 1, 1))

    sale_date = int(sale_date.strftime('%s')) // (60 * 60)


    if st.button("Predict"):
        payload = {
            "year": year,
            "make": make,
            "body": body,
            "transmission": transmission,
            "state": state,
            "condition": condition,
            "odometer": odometer,
            "color": color,
            "interior": interior,
            "mmr": mmr,
            "saledate": sale_date
        }

        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.write(f"Predicted Price: ${prediction:,.2f}")
        else:
            st.error(f"Error: {response.json()['detail']}")

if __name__ == "__main__":
    main()
