import streamlit as st
import pandas as pd
import numpy as np
from prediction import get_prediction, ordinal_encoder
import bz2file
import pickle

def decompress_pickle(file):
    data = bz2file.open(file, 'rb')
    return pickle.load(data)

model = decompress_pickle('Model/extra_tuned.pbz2')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

#"""options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
      # ' Industrial areas', 'School areas', '  Recreational areas',
       #' Outside rural areas', ' Hospital areas', '  Market areas',
       #'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       #'Recreational areas']"""
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
#options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
#"""options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
      # 'other', 'Double carriageway (median)', 'One way',
       #'Two-way (divided with solid lines road marking)', 'Unknown']"""
options_vehicle_owner = ['Owner', 'Governmental', 'Organization', 'Other']
options_junction_type = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other','Unknown', 'T Shape', 'X Shape']
options_surface_type = ['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress','Gravel roads', 'Other']
options_road_surface_conditions = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep']
options_light_condition = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit']
options_weather_condition = ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow', 'Unknown', 'Fog or mist']
features = ['day_of_week', 'driver_age', 'vehicle_type', 'vehicle_owner', 'junction_type', 'surface_type', 'road_surface_conditions', 'light_condition', 'weather_condition', 'vehicles_involved', 'casualties', 'accident_cause', 'hour', 'minute']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        minutes = st.slider("Pickup Minutes: ", 0, 59, value=0, format="%d")
        day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        driver_age = st.selectbox("Select Driver Age: ", options=options_age)
        vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        vehicle_owner = st.selectbox("Select Vehicle Owner: ", options=options_vehicle_owner)
        junction_type = st.selectbox("Select Junction Type: ", options=options_junction_type)
        surface_type = st.selectbox("Select Surface Type: ", options=options_surface_type)
        road_surface_conditions = st.selectbox("Select Road Surface Conditions: ", options=options_road_surface_conditions)
        weather_condition = st.selectbox("Select Weather Condition: ", options=options_weather_condition)
        light_condition = st.selectbox("Select Light Condition: ", options=options_light_condition)
        accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
        vehicles_involved = st.slider("Vehicle Involved: ", 1, 7, value=0, format="%d")
        casualties = st.slider("casualties: ", 1, 8, value=0, format="%d")
        #accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)
        #driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        #lanes = st.selectbox("Select Lanes: ", options=options_lanes)
        
        
        submit = st.form_submit_button("Predict")


    if submit:
        day_of_week = ordinal_encoder(day_of_week, options_day)
        accident_cause = ordinal_encoder(accident_cause, options_cause)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        driver_age =  ordinal_encoder(driver_age, options_age)
        vehicle_owner =  ordinal_encoder(vehicle_owner, options_vehicle_owner)
        junction_type = ordinal_encoder(junction_type, options_junction_type)
        surface_type = ordinal_encoder(surface_type, options_surface_type)
        road_surface_conditions = ordinal_encoder(road_surface_conditions, options_road_surface_conditions)
        light_condition = ordinal_encoder(light_condition, options_light_condition)
        weather_condition = ordinal_encoder(weather_condition, options_weather_condition)
        #accident_area =  ordinal_encoder(accident_area, options_acc_area)
        #driving_experience = ordinal_encoder(driving_experience, options_driver_exp) 
        #lanes = ordinal_encoder(lanes, options_lanes)


        data = np.array([day_of_week, driver_age, vehicle_type, vehicle_owner, junction_type, surface_type, road_surface_conditions,
                      light_condition, weather_condition, vehicles_involved, casualties, accident_cause, hour, minutes]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred}")

if __name__ == '__main__':
    main()