from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import streamlit as st

# =============================================================================
# Helper functions
# =============================================================================

def filter_and_clean_data(df):
    """
    Ensures 'Date' is datetime, filters out weekends, sorts by date,
    and converts Teacher_Mood to numeric.
    """
    # Convert Date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop rows where Date could not be converted
    df.dropna(subset=['Date'], inplace=True)

    # Filter out weekends
    df = df[df['Date'].dt.weekday < 5]

    # Sort by date
    df.sort_values('Date', inplace=True)

    # Convert Teacher_Mood to numeric
    df['Teacher_Mood'] = pd.to_numeric(df['Teacher_Mood'], errors='coerce')

    # Drop rows with null in Teacher_Mood
    df.dropna(subset=['Teacher_Mood'], inplace=True)

    return df


def train_teacher_mood_model(df):
    """
    Trains a logistic regression model to predict Teacher_Mood (1 = Chill, 0 = Crashout).
    """
    df = df.copy()
    # Check if there are at least two classes in Teacher_Mood
    unique_classes = df['Teacher_Mood'].unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Need at least two classes in 'Teacher_Mood'. Found: {unique_classes}")

    # Create a weekday column (0=Monday, 4=Friday)
    df['Weekday'] = df['Date'].dt.weekday
    # Compute the day index relative to the first data entry
    first_date = df['Date'].min()
    df['DayIndex'] = (df['Date'] - first_date).dt.days

    # One-hot encode the weekday (create wd_0, wd_1, …, wd_4)
    weekday_dummies = pd.get_dummies(df['Weekday'], prefix='wd')
    X = pd.concat([df['DayIndex'], weekday_dummies], axis=1)
    y = df['Teacher_Mood']

    model = LogisticRegression()
    model.fit(X, y)
    expected_features = X.columns  # e.g. ['DayIndex', 'wd_0', 'wd_1', ...]
    return model, expected_features, first_date


def predict_future_teacher_mood(model, expected_features, first_date, future_dates):
    """
    Predicts the teacher mood probabilities for a list of future dates.
    """
    predictions = []
    for date in future_dates:
        day_index = (date - first_date).days
        weekday = date.weekday()  # 0 (Mon) to 4 (Fri)
        # Build a dictionary with the features
        data = {'DayIndex': day_index}
        # Create one-hot encoding for weekdays 0-4
        for wd in range(5):
            data[f'wd_{wd}'] = 1 if weekday == wd else 0
        # Convert to DataFrame and reindex to ensure all expected features are present.
        df_features = pd.DataFrame([data])
        df_features = df_features.reindex(columns=expected_features, fill_value=0)
        prob = model.predict_proba(df_features)[0]
        predictions.append({
            'Date': date,
            'Prob_Crashout': prob[0],  # class 0 = Crashout
            'Prob_Chill': prob[1]      # class 1 = Chill
        })
    return pd.DataFrame(predictions)


# =============================================================================
# Streamlit Dashboard
# =============================================================================

st.set_page_config(page_title="Teacher Mood Prediction", layout="wide")
st.title("Teacher Mood Prediction Dashboard")
st.markdown("""
This dashboard predicts the probability that your teacher will **crash out** (bad mood) or **be chill** (good mood) tomorrow.  
You can input today's mood using the buttons below, and the app will update the predictions.
""")

# Initialize session state for storing data
if 'data' not in st.session_state:
    # Pre-populate with historical data containing both classes (0 and 1)
    historical_data = [
        {'Date': datetime(2023, 10, 1), 'Teacher_Mood': 1},
        {'Date': datetime(2023, 10, 2), 'Teacher_Mood': 0},
        {'Date': datetime(2023, 10, 3), 'Teacher_Mood': 1},
        {'Date': datetime(2023, 10, 4), 'Teacher_Mood': 0},
        {'Date': datetime(2023, 10, 5), 'Teacher_Mood': 1},
    ]
    st.session_state.data = pd.DataFrame(historical_data)

# Input today's mood
st.subheader("Input Today's Mood")
col1, col2 = st.columns(2)
with col1:
    if st.button("😊 Teacher was Chill today (1)"):
        today_mood = 1
        new_row = pd.DataFrame({'Date': [datetime.today()], 'Teacher_Mood': [today_mood]})
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        st.success("Today's mood (Chill) added to historical data!")
with col2:
    if st.button("😠 Teacher was Crashout today (0)"):
        today_mood = 0
        new_row = pd.DataFrame({'Date': [datetime.today()], 'Teacher_Mood': [today_mood]})
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        st.success("Today's mood (Crashout) added to historical data!")

# Clean and filter data
df = filter_and_clean_data(st.session_state.data)

# Check if there is enough data to train the model
if len(df) > 0:
    st.subheader("Historical Data")
    st.dataframe(df)

    # Check if there are at least two classes in Teacher_Mood
    unique_classes = df['Teacher_Mood'].unique()
    if len(unique_classes) < 2:
        st.warning(f"Need at least two classes in 'Teacher_Mood'. Found: {unique_classes}")
        st.warning("Please add more data with both 'Chill' and 'Crashout' moods to train the model.")
    else:
        # Train the model
        with st.spinner("Training prediction model..."):
            try:
                model, expected_features, first_date = train_teacher_mood_model(df)
                st.success("Model trained successfully!")

                # Predict for tomorrow
                tomorrow = datetime.today() + timedelta(days=1)
                if tomorrow.weekday() >= 5:  # Skip weekends
                    tomorrow = tomorrow + timedelta(days=(7 - tomorrow.weekday()))

                pred_tomorrow = predict_future_teacher_mood(model, expected_features, first_date, [tomorrow])

                # Display prediction for tomorrow
                st.subheader("Prediction for Tomorrow")
                tomorrow_date = pred_tomorrow.iloc[0]['Date'].strftime("%Y-%m-%d")
                tomorrow_crash_prob = pred_tomorrow.iloc[0]['Prob_Crashout']
                tomorrow_chill_prob = pred_tomorrow.iloc[0]['Prob_Chill']

                st.markdown(f"**Date:** {tomorrow_date}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Crashout Probability", value=f"{tomorrow_crash_prob * 100:.1f}%")
                with col2:
                    st.metric(label="Predicted Chill Probability", value=f"{tomorrow_chill_prob * 100:.1f}%")

                # Plot historical data and prediction
                st.subheader("Historical Moods and Prediction")
                fig, ax = plt.subplots(figsize=(10, 4))

                # Plot historical moods
                sns.scatterplot(
                    data=df,
                    x='Date',
                    y='Teacher_Mood',
                    hue='Teacher_Mood',
                    palette={0: "red", 1: "green"},
                    s=100,
                    ax=ax
                )

                # Plot tomorrow's prediction
                ax.scatter(
                    tomorrow,
                    0.5,  # Placeholder for prediction
                    color='blue',
                    s=200,
                    label='Prediction for Tomorrow'
                )

                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Crashout", "Chill"])
                ax.set_title("Historical Teacher Mood and Prediction for Tomorrow")
                ax.legend(loc='upper right')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("No data available. Please input today's mood using the buttons above.")