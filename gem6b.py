import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import emoji
import nltk
from nltk.corpus import stopwords
from fpdf import FPDF
import base64
from statsmodels.tsa.arima.model import ARIMA
import random
from textblob import TextBlob
from datetime import datetime, timedelta
import numpy as np




nltk.download("stopwords")
import bcrypt
import json
import os

import google.generativeai as genai

if "rando" in st.secrets:
    genai.configure(api_key=st.secrets["rando"])
else:
    st.warning(
        "Gemini  key not found in Streamlit. Some AI features may not work. Please add it  or set as an environment variable."
    )
    # For a quick local test (NOT for production or public code):
    # genai.configure(api_key="YOUR_GEMINI_API_KEY")


# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

CREDENTIALS_FILE = "users.json"


def load_users():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "r") as file:
            return json.load(file)
    return {}


def save_users(users):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(users, file)


def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = hashed_pw
    save_users(users)
    return True


def authenticate_user(username, password):
    users = load_users()
    if username in users and bcrypt.checkpw(
        password.encode(), users[username].encode()
    ):
        return True
    return False


# ===================== Preprocessing Function =====================
def preprocess_chat(data):
    data = data.replace("\u202f", " ").replace("\u200e", "")
    pattern = r"(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}) (am|pm) - (.*?): (.*)"
    matches = re.findall(pattern, data, flags=re.IGNORECASE)

    records = []
    media_messages_count = 0
    for date, time, ampm, user, message in matches:
        timestamp = f"{date} {time} {ampm}"
        try:
            dt = pd.to_datetime(timestamp, format="%d/%m/%Y %I:%M %p")
            if message.strip().lower() == "<media omitted>":
                media_messages_count += 1
            elif message.strip().lower() != "null":
                records.append((dt, user.strip(), message.strip()))
        except:
            continue

    df = pd.DataFrame(records, columns=["datetime", "user", "message"])
    df.loc[:, "media_message"] = (
        False  # Default to False for all records in the text dataframe
    )

    # Store media count separately, or add a conceptual row/metadata
    st.session_state["media_messages_count"] = media_messages_count

    return df


# ===================== Analysis Functions =====================
def filter_data(df, selected_users, start_date, end_date):
    if selected_users:
        df = df[df["user"].isin(selected_users)]
    if start_date and end_date:
        df = df[
            (df["datetime"] >= pd.to_datetime(start_date))
            & (df["datetime"] <= pd.to_datetime(end_date))
        ]
    return df


def weekly_activity(df):
    df["weekday"] = df["datetime"].dt.day_name()
    return df["weekday"].value_counts().sort_index()


def monthly_activity(df):
    df["month"] = df["datetime"].dt.strftime("%B %Y")
    return df["month"].value_counts().sort_index()


def hourly_distribution(df):
    df["hour"] = df["datetime"].dt.hour
    return df["hour"].value_counts().sort_index()


def heatmap_data(df):
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day_name()
    return df.pivot_table(
        index="day", columns="hour", values="message", aggfunc="count"
    ).fillna(0)


def common_words(df):
    stop_words = set(stopwords.words("english"))
    words = " ".join(df["message"]).lower()
    words = re.findall(r"\b\w+\b", words)
    filtered_words = [
        word for word in words if word not in stop_words and len(word) > 2
    ]
    return Counter(filtered_words).most_common(20)


def user_frequency(df):
    return df["user"].value_counts()


def generate_wordcloud(df):
    text = " ".join(df["message"])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wc


def emoji_counter(df):
    emojis = []
    for msg in df["message"]:
        emojis += [c for c in msg if c in emoji.EMOJI_DATA]
    return Counter(emojis).most_common(10)


def user_message_stats(df):
    df["length"] = df["message"].apply(len)
    avg_lengths = df.groupby("user")["length"].mean().sort_values(ascending=False)
    daily_counts = (
        df.groupby(["user", df["datetime"].dt.date]).size().groupby("user").mean()
    )
    return avg_lengths, daily_counts


def sentiment_analysis(df):
    # Ensure 'polarity' column is created within this function or before it's called
    df["polarity"] = df["message"].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    user_sentiment = df.groupby("user")["polarity"].mean().sort_values()
    return user_sentiment


def predict_busiest_month(df, steps=3):
    df["month"] = df["datetime"].dt.to_period("M")
    monthly_counts = df.groupby("month").size()
    monthly_counts.index = monthly_counts.index.to_timestamp()
    if len(monthly_counts) < 3:
        return None, None, "Not enough data for a trend."
    model = ARIMA(monthly_counts, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    predicted_month = forecast.idxmax()
    explanation = (
        f"The prediction is based on recent trends. "
        f"The forecast shows the highest expected activity in {predicted_month.strftime('%B %Y')} "
        f"with an estimated {int(forecast.max())} messages."
    )
    return predicted_month.strftime("%B %Y"), forecast, explanation


def generate_future_messages(df, num_sentences=5):
    text = " ".join(df["message"].dropna())
    words = text.split()
    if len(words) < 2:
        return ["Not enough data to predict future messages."]
    markov_chain = {}
    for i in range(len(words) - 1):
        key, next_word = words[i], words[i + 1]
        markov_chain.setdefault(key, []).append(next_word)
    messages = []
    for _ in range(num_sentences):
        word = random.choice(words)
        sentence = [word]
        for _ in range(random.randint(4, 12)):
            word = random.choice(markov_chain.get(word, words))
            sentence.append(word)
        messages.append(" ".join(sentence).capitalize() + ".")
    return messages


def predict_future_words(df, num_words=10):
    text = " ".join(df["message"].dropna()).lower()
    words = re.findall(r"\b\w+\b", text)
    if len(words) < 2:
        return ["Not enough data to predict future words."]

    markov_chain = {}
    for i in range(len(words) - 1):
        key = words[i]
        next_word = words[i + 1]
        markov_chain.setdefault(key, []).append(next_word)

    current_word = random.choice(words)
    future_words = [current_word]
    for _ in range(num_words - 1):
        next_words = markov_chain.get(current_word, words)
        current_word = random.choice(next_words)
        future_words.append(current_word)

    return future_words


def predict_active_days(df):
    weekday_counts = df["datetime"].dt.day_name().value_counts()
    top_days = weekday_counts.head(2).index.tolist()
    return (
        top_days,
        f"Based on past data, most messages are usually sent on {', '.join(top_days)}.",
    )


def predict_active_hours(df):
    hour_counts = df["datetime"].dt.hour.value_counts().sort_index()
    peak_hours = hour_counts[hour_counts == hour_counts.max()].index.tolist()
    return (
        peak_hours,
        f"Most activity has been observed around {', '.join(str(h) + ':00' for h in peak_hours)}.",
    )


def detect_message_trend(df):
    df["month"] = df["datetime"].dt.to_period("M")
    monthly = df.groupby("month").size()
    if len(monthly) < 3:
        return "Not enough data to detect trend."
    trend = "increasing" if monthly.iloc[-1] > monthly.iloc[0] else "decreasing"
    return f"Message volume is generally {trend} over time."


# New: Calculate Response Times
def calculate_response_times(df):
    df_sorted = df.sort_values(by="datetime").reset_index(drop=True)
    response_times = []

    # Iterate through messages to find responses from different users
    for i in range(1, len(df_sorted)):
        prev_msg = df_sorted.iloc[i - 1]
        current_msg = df_sorted.iloc[i]

        if current_msg["user"] != prev_msg["user"]:
            time_diff = (current_msg["datetime"] - prev_msg["datetime"]).total_seconds()
            response_times.append(time_diff)

    if not response_times:
        return None, "Not enough conversations to calculate response times."

    avg_response_sec = sum(response_times) / len(response_times)

    hours, remainder = divmod(avg_response_sec, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s", response_times


# New: Message Type Distribution
def message_type_distribution(total_text_messages, total_media_messages):
    labels = ["Text Messages", "Media Messages"]
    sizes = [total_text_messages, total_media_messages]

    if total_text_messages == 0 and total_media_messages == 0:
        return None, "No messages to display distribution."

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#66b3ff", "#99ff99"],
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig, None


# New: Top Active Hours List
def top_active_hours_list(df, top_k=5):
    hourly_counts = df["datetime"].dt.hour.value_counts().sort_values(ascending=False)
    if hourly_counts.empty:
        return "No activity data."

    top_hours = hourly_counts.head(top_k)
    return top_hours


# ===================== Gemini-powered features =====================


def summarize_chat(chat_text):
    if len(chat_text) > 10000:  # Limit input length for API call
        chat_text = chat_text[:10000] + "..."  # Truncate and indicate truncation
        st.warning(
            "Chat text too long for full summarization. Summarizing truncated text."
        )
    try:
        prompt = f"Summarize the following WhatsApp chat conversation. Focus on key topics and overall sentiment:\n\n{chat_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error summarizing chat: {e}"


def generate_smart_reply(last_message):
    try:
        prompt = f"Given the last message in a WhatsApp chat: '{last_message}', suggest a concise and relevant smart reply."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating smart reply: {e}"


# Modified: generate_message_content with length control, emoji option, and robust output cleaning
def generate_message_content(topic, tone="neutral", length="short", add_emojis=False):
    length_prompt = ""
    if length == "short":
        length_prompt = "Keep it concise and brief."
    elif length == "medium":
        length_prompt = "Aim for a moderate length, around 2-3 sentences."
    elif length == "long":
        length_prompt = "Make it detailed, around 4-5 sentences."
    elif length == "very long":
        length_prompt = "Elaborate extensively, providing comprehensive information."

    emoji_prompt = ""
    if add_emojis:
        emoji_prompt = (
            " Include relevant and expressive emojis to make it more vibrant."
        )

    try:
        # Strengthen the prompt to strictly output only the message
        prompt = f"Generate a WhatsApp message about '{topic}' in a '{tone}' tone. {length_prompt}{emoji_prompt} Output ONLY the message text. Do NOT include any explanations, introductions, or conversational filler."
        response = model.generate_content(prompt)

        # Post-process to remove any unwanted explanations or meta-text that the model might still add
        generated_text = response.text

        # Regex to remove common patterns of unwanted explanations
        # This will remove anything from "Explanation of choices" or "Explanation:" onwards
        # It also handles variations of "Best," or similar sign-offs followed by explanations
        generated_text = re.split(
            r"(\n+Explanation of choices|\n+Explanation:|\n*Best, .*?\n*Explanation of choices|\n*Best, .*?\n*Explanation:)",
            generated_text,
            1,
        )[0]
        generated_text = generated_text.strip()  # Remove leading/trailing whitespace

        # Also remove common conversational filler at the start/end if present
        if generated_text.lower().startswith("here's a "):
            generated_text = re.sub(
                r"^[Hh]ere's a (generated |suggested |concise )?whatsapp message( for you)?:?\s*",
                "",
                generated_text,
                flags=re.IGNORECASE,
            )
        if generated_text.lower().endswith("hope this helps."):
            generated_text = re.sub(
                r"hope this helps\.$", "", generated_text, flags=re.IGNORECASE
            )

        return generated_text.strip()
    except Exception as e:
        return f"Error generating message content: {e}"


def identify_key_topics(chat_text):
    if len(chat_text) > 10000:  # Limit input length for API call
        chat_text = chat_text[:10000] + "..."
        st.warning(
            "Chat text too long for full topic identification. Identifying topics from truncated text."
        )
    try:
        prompt = f"Identify the main topics or themes discussed in the following WhatsApp chat conversation. List them concisely.\n\n{chat_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error identifying topics: {e}"


# New: Function to generate a meeting summary
def generate_meeting_summary(chat_text):
    if len(chat_text) > 10000:
        chat_text = chat_text[:10000] + "..."
        st.warning(
            "Chat text too long for full meeting summarization. Summarizing truncated text."
        )
    try:
        prompt = f"Given the following chat, identify if it contains a meeting discussion. If so, summarize the key decisions, action items, and participants. If not, state that it does not appear to be a meeting chat:\n\n{chat_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating meeting summary: {e}"


# New: Function for sentiment trend over time
def sentiment_trend(df):
    df["date"] = df["datetime"].dt.date
    daily_sentiment = df.groupby("date")["polarity"].mean().reset_index()
    return daily_sentiment


# New: Function to identify most active hours by day of week
def active_hours_by_day(df):
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["hour"] = df["datetime"].dt.hour
    active_pivot = df.pivot_table(
        index="day_of_week", columns="hour", values="message", aggfunc="count"
    ).fillna(0)
    # Order days of week
    ordered_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    active_pivot = active_pivot.reindex(ordered_days, fill_value=0)
    return active_pivot


# New: Function to identify most common phrases
def common_phrases(df, n=2, top_k=10):
    text = " ".join(df["message"].dropna()).lower()
    words = re.findall(r"\b\w+\b", text)
    if len(words) < n:
        return []

    phrases = []
    for i in range(len(words) - n + 1):
        phrases.append(tuple(words[i : i + n]))

    return Counter(phrases).most_common(top_k)


# New: Function to identify recurring patterns/habits (basic example)
def identify_chat_habits(df):
    habits = []
    # Most active user
    most_active_user = df["user"].value_counts().idxmax()
    habits.append(f"Most active participant: **{most_active_user}**.")

    # Peak hour for messages
    peak_hour = df["datetime"].dt.hour.mode()[0]
    habits.append(f"Peak messaging hour: **{peak_hour}:00** to **{peak_hour+1}:00**.")

    # Most common day
    most_common_day = df["datetime"].dt.day_name().mode()[0]
    habits.append(f"Most active day of the week: **{most_common_day}**.")

    # Basic check for consistent morning/evening activity
    morning_msgs = df[(df["datetime"].dt.hour >= 6) & (df["datetime"].dt.hour < 12)]
    evening_msgs = df[(df["datetime"].dt.hour >= 18) & (df["datetime"].dt.hour < 24)]
    if len(morning_msgs) > len(df) * 0.3:
        habits.append("Consistent morning activity detected.")
    if len(evening_msgs) > len(df) * 0.3:
        habits.append("Consistent evening activity detected.")

    return habits


# New: Function to detect conversation starters
def detect_conversation_starters(df, top_k=5):
    starters = []
    # Using the first messages by each user on a given day/hour as potential starters
    df_sorted = df.sort_values(by="datetime")
    df_sorted["prev_user"] = df_sorted["user"].shift(1)
    df_sorted["time_diff_minutes"] = (
        df_sorted["datetime"] - df_sorted["datetime"].shift(1)
    ).dt.total_seconds() / 60

    potential_starters = df_sorted[
        (df_sorted["time_diff_minutes"].isna())  # First message
        | (df_sorted["time_diff_minutes"] > 30)  # Long pause (configurable threshold)
        | (
            df_sorted["datetime"].dt.date != df_sorted["datetime"].shift(1).dt.date
        )  # New day
    ]

    # Filter out media omitted or short/empty messages
    potential_starters = potential_starters[
        (potential_starters["message"].str.strip().str.len() > 5)
        & (~potential_starters["message"].str.contains(r"<media omitted>", case=False))
    ]

    if len(potential_starters) == 0:
        return ["Not enough distinct conversation starters detected."]

    return potential_starters["message"].value_counts().head(top_k).index.tolist()


# New: Analyze Overall Chat Tone/Mood
def analyze_overall_chat_tone(chat_text):
    if len(chat_text) > 10000:
        chat_text = chat_text[:10000] + "..."
        st.warning(
            "Chat text too long for full tone analysis. Analyzing truncated text."
        )
    try:
        prompt = f"Analyze the overall tone or mood of the following WhatsApp chat conversation. Is it generally positive, negative, neutral, humorous, formal, informal, serious, casual, etc.? Provide a concise description.\n\n{chat_text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing chat tone: {e}"


# NEW: Love Score Analysis (AI)
def calculate_love_score(chat_df):
    if chat_df.empty:
        return (
            "Not enough chat data to calculate a love score. Please upload a valid chat.",
            None,
            None,
            None,
        )

    # Get overall chat text
    full_chat_text = " ".join(chat_df["message"].dropna().tolist())

    # Calculate some basic metrics that can be included in the prompt
    num_users = chat_df["user"].nunique()
    total_messages = len(chat_df)

    if num_users != 2:
        return (
            f"This chat has {num_users} participants. For an accurate 'love score' based on couple dynamics, please upload a chat with only two participants.",
            None,
            None,
            None,
        )

    # Calculate sentiment
    chat_df["polarity"] = chat_df["message"].apply(
        lambda msg: TextBlob(msg).sentiment.polarity
    )
    avg_polarity = chat_df["polarity"].mean()

    # Communication balance
    user_message_counts = chat_df["user"].value_counts()
    user1, user2 = user_message_counts.index[0], user_message_counts.index[1]
    count1, count2 = user_message_counts.iloc[0], user_message_counts.iloc[1]
    balance = (
        min(count1, count2) / max(count1, count2) if max(count1, count2) > 0 else 0
    )

    # Common words (top 10 to include in prompt)
    common_w = common_words(chat_df)[:10]

    # Emojis (top 5 to include in prompt)
    common_e = emoji_counter(chat_df)[:5]

    # Length consideration for prompt
    prompt_chat_text = full_chat_text
    if (
        len(full_chat_text) > 8000
    ):  # Slightly larger limit for love score, but still cautious
        prompt_chat_text = full_chat_text[:8000] + "..."  # Truncate for prompt
        st.warning(
            "Chat text too long for full love score analysis. Analyzing truncated text."
        )

    # Constructing a detailed prompt for Gemini
    prompt = f"""
    You are an AI relationship analyst. Analyze the following WhatsApp chat conversation between two individuals. 
    Your goal is to assess the relationship's "love score" on a scale of 1 to 100, where 100 is highly loving and connected, and 1 is distant or problematic.

    Consider the following aspects of the chat:
    - **Overall Sentiment:** The general emotional tone of messages. (Average polarity: {avg_polarity:.2f})
    - **Communication Balance:** How evenly do the two participants contribute to the conversation? (Message ratio of {user1}: {count1} to {user2}: {count2}. Balance score: {balance:.2f})
    - **Frequency of Communication:** How consistently do they interact? (Total messages: {total_messages})
    - **Use of Affectionate Language/Emojis:** Presence of terms of endearment, positive words, and love-related emojis. (Common words: {common_w}, Common emojis: {common_e})
    - **Shared Interests/Topics:** Do they discuss common topics or support each other's interests?
    - **Empathy and Support:** Do messages show understanding, care, and mutual encouragement?
    - **Conflict Resolution (if apparent):** How are disagreements handled? (If any are present in the provided text.)
    - **Engagement:** Are responses timely and thoughtful, or short and dismissive?

    The chat conversation is as follows:
    ---
    {prompt_chat_text}
    ---

    Based on your analysis, provide:
    1.  **A Love Score (out of 100):** A numerical value.
    2.  **A Detailed Explanation:** Justify your score by explicitly referencing the positive and negative aspects observed in the chat, based on the criteria above. Be specific.
    3.  **Recommendations (Optional):** Suggest areas for improvement or ways to foster an even stronger connection, if applicable.

    Format your output clearly:
    Love Score: [SCORE]/100
    Explanation: [Your detailed explanation]
    Recommendations: [Your recommendations or "N/A"]
    """

    try:
        response = model.generate_content(prompt)
        # Attempt to parse the response
        response_text = response.text

        score_match = re.search(
            r"Love Score:\s*(\d+)/100", response_text, re.IGNORECASE
        )
        score = int(score_match.group(1)) if score_match else None

        explanation_match = re.search(
            r"Explanation:\s*(.*?)(?=\nRecommendations:|\nLove Score:|$)",
            response_text,
            re.IGNORECASE | re.DOTALL,
        )
        explanation = (
            explanation_match.group(1).strip()
            if explanation_match
            else "Could not extract explanation."
        )

        recommendations_match = re.search(
            r"Recommendations:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL
        )
        recommendations = (
            recommendations_match.group(1).strip() if recommendations_match else "N/A"
        )

        return "Success", score, explanation, recommendations

    except Exception as e:
        return f"Error calculating love score: {e}", None, None, None


# NEW FEATURES FOR LOVE SCORE SECTION


def calculate_relationship_timeline_insights(df):
    if df.empty:
        return None, None, None, None

    first_msg_date = df["datetime"].min().date()
    last_msg_date = df["datetime"].max().date()

    # Corrected variable name: days_active instead of total_days_chatted
    days_active = (last_msg_date - first_msg_date).days + 1

    # Longest gap between replies (between different users)
    df_sorted = df.sort_values(by="datetime").reset_index(drop=True)
    longest_gap = timedelta(0)

    # Iterate through messages to find responses from different users
    if len(df_sorted) > 1:
        for i in range(1, len(df_sorted)):
            prev_msg = df_sorted.iloc[i - 1]
            current_msg = df_sorted.iloc[i]

            if current_msg["user"] != prev_msg["user"]:
                time_diff = current_msg["datetime"] - prev_msg["datetime"]
                if time_diff > longest_gap:
                    longest_gap = time_diff

    # Streaks of consistent daily chatting
    active_dates = df["datetime"].dt.date.unique()
    active_dates.sort()

    longest_streak = 0
    current_streak = 0
    if len(active_dates) > 0:
        current_streak = 1
        longest_streak = 1
        for i in range(1, len(active_dates)):
            if (active_dates[i] - active_dates[i - 1]).days == 1:
                current_streak += 1
            else:
                current_streak = 1
            longest_streak = max(longest_streak, current_streak)

    return first_msg_date, days_active, longest_gap, longest_streak


def get_conversation_balance(df):
    if df.empty:
        return None, None, None, None, None

    user_message_counts = df["user"].value_counts()

    # Chat dominance ratio
    if len(user_message_counts) == 2:
        user1_name, user2_name = (
            user_message_counts.index[0],
            user_message_counts.index[1],
        )
        user1_count, user2_count = (
            user_message_counts.iloc[0],
            user_message_counts.iloc[1],
        )

        ratio = f"{user1_name}: {user1_count}, {user2_name}: {user2_count}"
        dominance_text = f"The message distribution is {user1_name} ({user1_count} messages) and {user2_name} ({user2_count} messages)."

        # Who sends longer messages
        df["length"] = df["message"].apply(len)
        avg_lengths = df.groupby("user")["length"].mean()
        longer_msgs_user = avg_lengths.idxmax()
        longer_msgs_avg = avg_lengths.max()

        # Who initiates more messages (first message after a 30 min idle period or new day)
        df_sorted = df.sort_values(by="datetime").reset_index(drop=True)
        initiator_counts = Counter()

        if not df_sorted.empty:
            initiator_counts[df_sorted.iloc[0]["user"]] += 1  # First message overall
            for i in range(1, len(df_sorted)):
                time_diff = (
                    df_sorted.iloc[i]["datetime"] - df_sorted.iloc[i - 1]["datetime"]
                ).total_seconds() / 60  # in minutes
                if (
                    time_diff > 30
                    or df_sorted.iloc[i]["datetime"].date()
                    != df_sorted.iloc[i - 1]["datetime"].date()
                ):
                    initiator_counts[df_sorted.iloc[i]["user"]] += 1

        if initiator_counts:
            most_initiator = initiator_counts.most_common(1)[0][0]
            initiator_text = (
                f"**{most_initiator}** seems to initiate conversations more often."
            )
        else:
            initiator_text = "Could not determine conversation initiators."

        return (
            dominance_text,
            longer_msgs_user,
            longer_msgs_avg,
            initiator_text,
            user_message_counts,
        )
    else:
        return (
            "Not a two-person chat.",
            None,
            None,
            None,
            user_message_counts,
        )  # Return user_message_counts even if not 2 people


def top_romantic_messages(df, num_messages=5):
    if df.empty:
        return pd.DataFrame()

    df["polarity"] = df["message"].apply(lambda x: TextBlob(x).sentiment.polarity)
    most_romantic = df[df["polarity"] > 0.3].sort_values(by="polarity", ascending=False)

    # Filter out short messages or media omitted placeholders
    most_romantic = most_romantic[most_romantic["message"].str.strip().str.len() > 10]
    most_romantic = most_romantic[
        ~most_romantic["message"].str.contains(r"<media omitted>", case=False)
    ]

    return most_romantic[["datetime", "user", "message", "polarity"]].head(num_messages)


def calculate_response_times_per_user(df):
    if df.empty or df["user"].nunique() < 2:
        return "Not enough data for per-user response times.", {}, None

    df_sorted = df.sort_values(by="datetime").reset_index(drop=True)
    user_response_times = {user: [] for user in df["user"].unique()}

    # Map user to their last message time
    last_message_time = {user: None for user in df["user"].unique()}

    for i in range(len(df_sorted)):
        current_msg = df_sorted.iloc[i]
        current_user = current_msg["user"]
        current_time = current_msg["datetime"]

        # Update last message time for the current user
        last_message_time[current_user] = current_time

        # To calculate current_user's response time to previous message (from other user)
        if (
            i > 0 and df_sorted.iloc[i - 1]["user"] != current_user
        ):  # If the previous message was from the *other* user
            response_diff = current_time - df_sorted.iloc[i - 1]["datetime"]
            user_response_times[current_user].append(response_diff.total_seconds())

    avg_response_sec_per_user = {
        user: (sum(times) / len(times)) if times else 0
        for user, times in user_response_times.items()
    }

    # Convert to readable format
    avg_response_readable = {}
    for user, seconds in avg_response_sec_per_user.items():
        if seconds > 0:
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds_rem = divmod(remainder, 60)
            avg_response_readable[user] = (
                f"{int(hours)}h {int(minutes)}m {int(seconds_rem)}s"
            )
        else:
            avg_response_readable[user] = "N/A (No responses detected)"

    fastest_replier = None
    min_avg_time = float("inf")
    for user, avg_sec in avg_response_sec_per_user.items():
        if avg_sec > 0 and avg_sec < min_avg_time:
            min_avg_time = avg_sec
            fastest_replier = user

    return avg_response_readable, avg_response_sec_per_user, fastest_replier


# New: Personality Tags (AI-based Guess) - Renamed to guess_personality_traits for MBTI-like
def guess_personality_traits(chat_df):
    if chat_df.empty or chat_df["user"].nunique() != 2:
        return "Please upload a two-person chat for personality insights."

    user_stats = {}

    for user in chat_df["user"].unique():
        user_df = chat_df[chat_df["user"] == user].copy()
        user_text = " ".join(user_df["message"].dropna().tolist())

        # Calculate user-specific metrics
        num_messages = len(user_df)
        avg_sentiment = (
            user_df["message"]
            .apply(lambda msg: TextBlob(msg).sentiment.polarity)
            .mean()
        )
        avg_length = user_df["message"].apply(len).mean()

        common_w = ", ".join([word for word, count in common_words(user_df)[:5]])
        common_e = ", ".join([e for e, count in emoji_counter(user_df)[:3]])

        user_stats[user] = {
            "num_messages": num_messages,
            "avg_sentiment": avg_sentiment,
            "avg_length": avg_length,
            "common_words": common_w,
            "common_emojis": common_e,
            "sample_messages": " ".join(
                user_df["message"].sample(min(5, len(user_df))).tolist()
            ),
        }

    user1_name, user2_name = list(user_stats.keys())
    user1_stats = user_stats[user1_name]
    user2_stats = user_stats[user2_name]

    prompt = f"""
    You are an AI personality profiler. Analyze the messaging styles of two individuals in a WhatsApp chat.
    Based on the provided statistics and sample messages, infer potential personality traits for each person, similar to a light MBTI-like guess (e.g., "warm and expressive", "reserved and analytical", "spontaneous and energetic").
    You can also suggest a relevant emoji or symbol for their style.

    Here are the statistics for {user1_name}:
    - Total Messages: {user1_stats['num_messages']}
    - Avg. Sentiment: {user1_stats['avg_sentiment']:.2f}
    - Avg. Message Length: {user1_stats['avg_length']:.2f} characters
    - Common Words: {user1_stats['common_words']}
    - Common Emojis: {user1_stats['common_emojis']}
    - Sample Messages: "{user1_stats['sample_messages']}"

    Here are the statistics for {user2_name}:
    - Total Messages: {user2_stats['num_messages']}
    - Avg. Sentiment: {user2_stats['avg_sentiment']:.2f}
    - Avg. Message Length: {user2_stats['avg_length']:.2f} characters
    - Common Words: {user2_stats['common_words']}
    - Common Emojis: {user2_stats['common_emojis']}
    - Sample Messages: "{user2_stats['sample_messages']}"

    Based on this, provide a personality description for each user.
    Format your output as:
    **{user1_name}:** [Personality Description with emoji]
    **{user2_name}:** [Personality Description with emoji]
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error guessing personality traits: {e}"


# New: Conflict & Recovery Patterns
def detect_conflict_and_recovery(
    df, time_window_minutes=60, sentiment_threshold=-0.1, recovery_threshold=0.2
):
    if df.empty or df["user"].nunique() < 2:
        return "Not enough data for conflict detection or not a two-person chat."

    df["polarity"] = df["message"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_sorted = df.sort_values(by="datetime").reset_index(drop=True)

    conflict_patterns = []

    for i in range(len(df_sorted)):
        current_msg = df_sorted.iloc[i]

        # Look for negative sentiment
        if current_msg["polarity"] < sentiment_threshold:
            # Define a time window after this negative message
            end_window_time = current_msg["datetime"] + timedelta(
                minutes=time_window_minutes
            )

            # Check for subsequent messages within the window
            subsequent_msgs = df_sorted[
                (df_sorted["datetime"] > current_msg["datetime"])
                & (df_sorted["datetime"] <= end_window_time)
                & (
                    df_sorted.index > i
                )  # Ensure we look forward from the current message
            ]

            if not subsequent_msgs.empty:
                # Check for significant positive shift within the window
                max_pos_polarity_in_window = subsequent_msgs["polarity"].max()

                if (
                    max_pos_polarity_in_window > recovery_threshold
                    and (max_pos_polarity_in_window - current_msg["polarity"])
                    > recovery_threshold
                ):
                    # Found a potential recovery
                    recovery_msg = subsequent_msgs.loc[
                        subsequent_msgs["polarity"].idxmax()
                    ]

                    conflict_patterns.append(
                        {
                            "start_time": current_msg["datetime"],
                            "end_time": recovery_msg["datetime"],
                            "conflict_message": current_msg["message"],
                            "recovery_message": recovery_msg["message"],
                            "time_to_recover": recovery_msg["datetime"]
                            - current_msg["datetime"],
                        }
                    )

    if conflict_patterns:
        result = "Detected potential conflict and recovery patterns:\n"
        for pattern in conflict_patterns:
            result += f"- Conflict identified around {pattern['start_time'].strftime('%Y-%m-%d %H:%M')} with message: \"{pattern['conflict_message']}\"\n"
            result += f"  Recovery message around {pattern['end_time'].strftime('%Y-%m-%d %H:%M')} with message: \"{pattern['recovery_message']}\"\n"
            result += f"  Time to recovery: {pattern['time_to_recover']}\n\n"
        return result
    else:
        return "No clear conflict and recovery patterns detected based on sentiment swings."


# New: Compatibility Radar Chart
def create_compatibility_radar_chart(df):
    if df.empty or df["user"].nunique() != 2:
        return (
            None,
            "Please upload a two-person chat to generate a compatibility radar chart.",
        )

    users = df["user"].unique()
    if len(users) != 2:  # Redundant check but good for safety
        return (
            None,
            "Radar chart is best for comparing two users. This chat has more/less than two participants.",
        )

    user1, user2 = users[0], users[1]

    # Calculate metrics for each user
    df["polarity"] = df["message"].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    df["length"] = df["message"].apply(len)

    metrics = {}
    for user in users:
        user_df = df[df["user"] == user]
        metrics[user] = {
            "avg_sentiment": user_df["polarity"].mean(),
            "msg_frequency": len(user_df),
            "avg_msg_length": user_df["length"].mean(),
            "avg_daily_messages": (
                user_df.groupby(user_df["datetime"].dt.date).size().mean()
                if not user_df.empty
                else 0
            ),
        }

    # Normalize metrics for radar chart (0 to 1 scale)
    # Define max values for normalization - these can be adjusted based on typical chat data
    # Use max values from *both* users or a reasonable global max
    all_msg_frequencies = [metrics[u]["msg_frequency"] for u in users]
    all_avg_msg_lengths = [metrics[u]["avg_msg_length"] for u in users]
    all_avg_daily_messages = [metrics[u]["avg_daily_messages"] for u in users]

    max_msg_frequency = (
        max(all_msg_frequencies) * 1.2 if all_msg_frequencies else 1
    )  # Add buffer
    max_avg_msg_length = max(all_avg_msg_lengths) * 1.2 if all_avg_msg_lengths else 1
    max_avg_daily_messages = (
        max(all_avg_daily_messages) * 1.2 if all_avg_daily_messages else 1
    )

    radar_values_user1 = []
    radar_values_user2 = []

    # Axes for Radar Chart
    labels = ["Positivity", "Message Frequency", "Message Length", "Daily Consistency"]

    # Positivity (already -1 to 1, normalize to 0-1)
    radar_values_user1.append((metrics[user1]["avg_sentiment"] + 1) / 2)
    radar_values_user2.append((metrics[user2]["avg_sentiment"] + 1) / 2)

    # Message Frequency (normalized by max frequency in chat)
    radar_values_user1.append(metrics[user1]["msg_frequency"] / max_msg_frequency)
    radar_values_user2.append(metrics[user2]["msg_frequency"] / max_msg_frequency)

    # Message Length (normalized by overall max msg length)
    radar_values_user1.append(metrics[user1]["avg_msg_length"] / max_avg_msg_length)
    radar_values_user2.append(metrics[user2]["avg_msg_length"] / max_avg_msg_length)

    # Daily Consistency (normalized by max avg daily messages)
    radar_values_user1.append(
        metrics[user1]["avg_daily_messages"] / max_avg_daily_messages
    )
    radar_values_user2.append(
        metrics[user2]["avg_daily_messages"] / max_avg_daily_messages
    )

    # Make the plot circular
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]  # Use np.pi
    angles += angles[:1]  # Complete the circle

    radar_values_user1 += radar_values_user1[:1]
    radar_values_user2 += radar_values_user2[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(
        angles,
        radar_values_user1,
        linewidth=2,
        linestyle="solid",
        label=user1,
        color="blue",
        alpha=0.7,
    )
    ax.fill(angles, radar_values_user1, color="blue", alpha=0.25)

    ax.plot(
        angles,
        radar_values_user2,
        linewidth=2,
        linestyle="solid",
        label=user2,
        color="red",
        alpha=0.7,
    )
    ax.fill(angles, radar_values_user2, color="red", alpha=0.25)

    ax.set_theta_offset(np.pi / 2)  # Use np.pi
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)  # Normalized scale
    ax.set_title("Relationship Compatibility Radar", size=16, color="grey", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    return fig, None


# NEW RELATIONSHIP FEATURES FUNCTIONS


# 1. Emotional Compatibility Meter
def emotional_compatibility_meter(df):
    if df.empty or df["user"].nunique() != 2:
        return "Please upload a two-person chat for emotional compatibility analysis."

    df["polarity"] = df["message"].apply(lambda msg: TextBlob(msg).sentiment.polarity)

    users = df["user"].unique()
    user1, user2 = users[0], users[1]

    user1_sentiment = df[df["user"] == user1]["polarity"]
    user2_sentiment = df[df["user"] == user2]["polarity"]

    avg_sentiment1 = user1_sentiment.mean()
    avg_sentiment2 = user2_sentiment.mean()

    std_sentiment1 = user1_sentiment.std()
    std_sentiment2 = user2_sentiment.std()

    # Calculate emotional sync score (higher is better)
    # 1. Similarity in average sentiment (closer to 0 difference is better)
    avg_sentiment_diff = abs(avg_sentiment1 - avg_sentiment2)
    # Normalize difference: max diff is 2 (1 to -1), so (2 - diff)/2
    avg_sentiment_sync = (2 - avg_sentiment_diff) / 2

    # 2. Similarity in emotional range (closer to 0 difference is better)
    # Handle cases where std is NaN (e.g., very few messages)
    std_sentiment_diff = (
        abs(std_sentiment1 - std_sentiment2)
        if not (pd.isna(std_sentiment1) or pd.isna(std_sentiment2))
        else 0
    )
    # Normalize std dev diff: max std dev is 1, so (1 - diff)/1
    std_sentiment_sync = (
        (1 - std_sentiment_diff) if std_sentiment_diff <= 1 else 0
    )  # Cap at 0 if diff > 1

    # Combine scores (weighted, can be adjusted)
    emotional_sync_score = (
        avg_sentiment_sync * 0.6 + std_sentiment_sync * 0.4
    ) * 100  # Scale to 100

    interpretation = ""
    if emotional_sync_score >= 80:
        interpretation = "You both ride the same emotional wave 🌊! Highly in sync."
    elif emotional_sync_score >= 60:
        interpretation = "Good emotional compatibility, generally on the same page. 👍"
    elif emotional_sync_score >= 40:
        interpretation = "Moderate emotional compatibility. Some differences in emotional expression."
    else:
        interpretation = "There might be some differences in how you express or experience emotions. Consider open communication. 💬"

    return (
        emotional_sync_score,
        interpretation,
        avg_sentiment1,
        avg_sentiment2,
        std_sentiment1,
        std_sentiment2,
    )


# 2. Message Type Classifier (Heuristic)
def classify_message_types(df):
    if df.empty:
        return "No messages to classify.", None

    message_types = {
        "Romantic 💘": [
            "love",
            "miss you",
            "babe",
            "honey",
            "darling",
            "sweetheart",
            "xoxo",
            "kiss",
            "hug",
            "cutie",
            "my dear",
            "❤️",
            "🥰",
            "😘",
            "😍",
            "💕",
            "💖",
            "💞",
        ],
        "Conversational 🗣️": [
            "hey",
            "hi",
            "how are you",
            "what's up",
            "lol",
            "haha",
            "yeah",
            "okay",
            "alright",
            "just checking in",
            "chat soon",
        ],
        "Curious 🤔": [
            "?",
            "what",
            "when",
            "where",
            "why",
            "how",
            "tell me more",
            "curious",
            "wondering",
        ],
        "Intellectual 🧠": [
            "think about",
            "idea",
            "concept",
            "theory",
            "article",
            "research",
            "analysis",
            "discuss",
            "perspective",
        ],
        "Needy 🥺": [
            "need you",
            "lonely",
            "miss you so much",
            "please",
            "can you",
            "are you there",
            "waiting",
            "🥺",
            "😔",
            "😭",
        ],
    }

    message_counts = {k: 0 for k in message_types.keys()}

    for message in df["message"].dropna().str.lower():
        assigned = False
        for msg_type, keywords in message_types.items():
            if any(keyword in message for keyword in keywords):
                message_counts[msg_type] += 1
                assigned = True
                break  # Assign to the first matching category
        # If no specific category matches, it's just general conversation
        # We could add a 'General' category, but for now, it's implicitly not counted in the above.

    total_classified = sum(message_counts.values())
    if total_classified == 0:
        return "No messages matched defined categories.", None

    # Filter out types with 0 messages for the pie chart
    filtered_counts = {k: v for k, v in message_counts.items() if v > 0}

    labels = filtered_counts.keys()
    sizes = filtered_counts.values()

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "black"},
    )
    ax.axis("equal")
    ax.set_title("Distribution of Message Types")
    return "Message type distribution:", fig


# 3. Relationship Diary Generator (Auto-TL;DR)
def generate_relationship_diary(df):
    if df.empty or df["user"].nunique() != 2:
        return "Please upload a two-person chat for the relationship diary."

    first_msg_date, total_days_chatted, longest_gap, longest_streak = (
        calculate_relationship_timeline_insights(df)
    )

    df["polarity"] = df["message"].apply(lambda msg: TextBlob(msg).sentiment.polarity)
    avg_sentiment = df["polarity"].mean()

    user_message_counts = df["user"].value_counts()
    user1, user2 = user_message_counts.index[0], user_message_counts.index[1]
    count1, count2 = user_message_counts.iloc[0], user_message_counts.iloc[1]

    # Get top emojis
    emojis = emoji_counter(df)
    top_emojis_str = (
        ", ".join([f"{e[0]} ({e[1]})" for e in emojis[:3]]) if emojis else "none"
    )

    # Get response times
    avg_res_readable, _, fastest_replier = calculate_response_times_per_user(df)
    response_time_info = ""
    if fastest_replier:
        response_time_info = f"{fastest_replier} tends to reply faster."
    else:
        response_time_info = "Response time patterns are balanced or unclear."

    # Count "I love you" mentions (simple keyword search)
    love_you_count = df["message"].str.lower().str.count(r"i love you").sum()

    chat_summary_text = " ".join(df["message"].dropna().tolist())
    if len(chat_summary_text) > 5000:  # Truncate for AI prompt
        chat_summary_text = chat_summary_text[:5000] + "..."

    prompt = f"""
    Write a concise and warm relationship diary entry summarizing the communication between two partners based on the following data:
    - Chat start date: {first_msg_date}
    - Total days chatted: {total_days_chatted}
    - Total messages: {len(df)}
    - Average chat sentiment (polarity -1 to 1): {avg_sentiment:.2f}
    - Message count: {user1} sent {count1} messages, {user2} sent {count2} messages.
    - Top 3 emojis used: {top_emojis_str}
    - "I love you" mentions: {love_you_count} times
    - Response time insight: {response_time_info}
    - Sample chat content: {chat_summary_text}

    Focus on the overall mood, key communication patterns, and highlights. Make it sound like a personal diary entry.
    Example: "March was filled with warm moments. You exchanged 437 messages, said 'I love you' 13 times, and used 🥺 most often. Aisha sent more messages, but you replied faster."
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating relationship diary: {e}"


# 4. Attachment Style Prediction (Experimental - Rule-based)
def predict_attachment_style(df):
    if df.empty or df["user"].nunique() != 2:
        return "Please upload a two-person chat for attachment style prediction."

    df["polarity"] = df["message"].apply(lambda msg: TextBlob(msg).sentiment.polarity)

    users = df["user"].unique()
    user_styles = {}

    avg_res_readable, avg_res_sec_per_user, fastest_replier = (
        calculate_response_times_per_user(df)
    )

    for user in users:
        user_df = df[df["user"] == user]
        num_messages = len(user_df)
        avg_sentiment = user_df["polarity"].mean()

        # Calculate response time for this user (how fast they reply to others)
        response_time_seconds = avg_res_sec_per_user.get(user, 0)

        # Heuristic rules for attachment styles
        style = "Unknown"
        explanation = []

        # Secure: Balanced, positive, moderate response
        if (
            (num_messages > len(df) * 0.4 and num_messages < len(df) * 0.6)
            and avg_sentiment > 0.1
            and (response_time_seconds > 60 and response_time_seconds < 3600)
        ):  # 1 min to 1 hour
            style = "Secure 💚"
            explanation.append(
                "Balanced communication, generally positive sentiment, and healthy response times."
            )

        # Anxious: High frequency, quick response, sometimes lower sentiment if not reciprocated
        elif (
            num_messages > len(df) * 0.6
            and avg_sentiment > -0.1
            and response_time_seconds > 0
            and response_time_seconds < 300
        ):  # Less than 5 minutes, ensure response_time_seconds is not 0
            style = "Anxious 🧡"
            explanation.append(
                "High message frequency and quick responses, potentially indicating a need for reassurance."
            )

        # Avoidant: Lower frequency, longer response, neutral/negative sentiment
        elif (
            num_messages < len(df) * 0.3
            and avg_sentiment < 0.1
            and response_time_seconds > 3600 * 2
        ):  # More than 2 hours
            style = "Avoidant 💙"
            explanation.append(
                "Lower message frequency and longer response times, possibly indicating a preference for independence."
            )

        # Fallback for other patterns
        if style == "Unknown":
            if num_messages > len(df) * 0.7:
                style = "Highly Engaged"
                explanation.append(
                    "Very high message frequency, indicating strong engagement."
                )
            elif num_messages < len(df) * 0.3:
                style = "Less Engaged"
                explanation.append(
                    "Lower message frequency, suggesting less active participation."
                )

            if avg_sentiment < -0.2:
                explanation.append("Tends to have a more negative sentiment.")
            elif avg_sentiment > 0.2:
                explanation.append("Tends to have a more positive sentiment.")

        user_styles[user] = {
            "style": style,
            "explanation": (
                " ".join(explanation)
                if explanation
                else "General communication patterns."
            ),
        }

    return user_styles


# 5. Mood Shifts Over Time (already handled by sentiment_trend function, just need to display it)


# 6. Love Language Classifier (Heuristic)
def classify_love_language(df):
    if df.empty:
        return "No messages to classify love languages.", None

    love_languages = {
        "Words of Affirmation 💬": [
            "i love you",
            "i appreciate you",
            "thank you",
            "you are the best",
            "amazing",
            "great job",
            "so proud",
            "miss you",
            "thinking of you",
            "beautiful",
            "handsome",
            "sweet",
        ],
        "Quality Time ⏰": [
            "let's call",
            "call me",
            "meet up",
            "hang out",
            "see you soon",
            "spend time",
            "date night",
            "together",
            "facetime",
            "video call",
        ],
        "Acts of Service 🛠️": [
            "i can help",
            "let me know if you need",
            "did you eat",
            "take care",
            "i'll do it",
            "don't worry",
            "i got this",
            "done for you",
            "how can i help",
        ],
        "Receiving Gifts 🎁": [
            "gift",
            "present",
            "bought you",
            "found this for you",
            "surprise",
            "got you",
            "treat",
        ],
        "Physical Touch (implied) 🤗": [
            "hug",
            "cuddle",
            "kiss",
            "touch",
            "snuggle",
            "come here",
            "miss your touch",
            "🤗",
            "😘",
            "🥰",
        ],  # More implied for chat
    }

    language_counts = {k: 0 for k in love_languages.keys()}

    for message in df["message"].dropna().str.lower():
        for lang, keywords in love_languages.items():
            if any(keyword in message for keyword in keywords):
                language_counts[lang] += 1

    total_classified = sum(language_counts.values())
    if total_classified == 0:
        return "No messages matched defined love language categories.", None

    filtered_counts = {k: v for k, v in language_counts.items() if v > 0}

    labels = filtered_counts.keys()
    sizes = filtered_counts.values()

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "black"},
    )
    ax.axis("equal")
    ax.set_title("Dominant Love Languages (Based on Chat Content)")
    return "Love language distribution:", fig


# 7. Personality Traits (Light MBTI Guess) - using existing guess_personality_traits


# ===================== BUSINESS FEATURES FUNCTIONS =====================


# 1. Response Time Insights (adapted from existing, focusing on business context)
def business_response_time_insights(df):
    if df.empty:
        return "No chat data available for response time analysis."

    df_sorted = df.sort_values(by="datetime").reset_index(drop=True)
    response_times_seconds = []

    # To identify fastest/slowest replies, we need to track individual response times
    individual_response_times = []  # List of (user, time_diff_seconds)

    # Map user to their last message time
    last_message_time = {user: None for user in df["user"].unique()}

    for i in range(len(df_sorted)):
        current_msg = df_sorted.iloc[i]
        current_user = current_msg["user"]
        current_time = current_msg["datetime"]

        # Check if the previous message was from a different user
        if i > 0 and df_sorted.iloc[i - 1]["user"] != current_user:
            prev_msg_time = df_sorted.iloc[i - 1]["datetime"]
            time_diff = (current_time - prev_msg_time).total_seconds()
            response_times_seconds.append(time_diff)
            individual_response_times.append(
                {
                    "user": current_user,
                    "time_diff": time_diff,
                    "message": current_msg["message"],
                }
            )

        # Update last message time for the current user
        last_message_time[current_user] = current_time

    if not response_times_seconds:
        return "Not enough back-and-forth messages between different users to calculate response times."

    avg_response_sec = sum(response_times_seconds) / len(response_times_seconds)

    hours, remainder = divmod(avg_response_sec, 3600)
    minutes, seconds = divmod(remainder, 60)
    avg_response_readable = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    # Fastest and slowest reply cases
    fastest_reply = None
    slowest_reply = None
    if individual_response_times:
        fastest_reply = min(individual_response_times, key=lambda x: x["time_diff"])
        slowest_reply = max(individual_response_times, key=lambda x: x["time_diff"])

    result_str = f"**Overall Average Response Time:** {avg_response_readable}\n\n"

    if fastest_reply:
        fastest_h, fastest_m, fastest_s = (
            int(fastest_reply["time_diff"] // 3600),
            int((fastest_reply["time_diff"] % 3600) // 60),
            int(fastest_reply["time_diff"] % 60),
        )
        result_str += f"**Fastest Reply:** by {fastest_reply['user']} in {fastest_h}h {fastest_m}m {fastest_s}s. Message: \"{fastest_reply['message'][:50]}...\"\n"
    if slowest_reply:
        slowest_h, slowest_m, slowest_s = (
            int(slowest_reply["time_diff"] // 3600),
            int((slowest_reply["time_diff"] % 3600) // 60),
            int(slowest_reply["time_diff"] % 60),
        )
        result_str += f"**Slowest Reply:** by {slowest_reply['user']} in {slowest_h}h {slowest_m}m {slowest_s}s. Message: \"{slowest_reply['message'][:50]}...\"\n"

    # Plot distribution
    fig, ax = plt.subplots()
    # Convert to minutes for plotting if values are large
    response_times_minutes = [t / 60 for t in response_times_seconds]
    sns.histplot(response_times_minutes, bins=20, kde=True, ax=ax)
    ax.set_title("Distribution of Response Times (Minutes)")
    ax.set_xlabel("Response Time (Minutes)")
    ax.set_ylabel("Frequency")

    return result_str, fig


# 2. Order Tracker (Simplified)
def order_tracker(df):
    if df.empty:
        return "No chat data to track orders."

    order_keywords = {
        "Order Placed 📝": [
            "order placed",
            "placed an order",
            "booked",
            "confirm order",
            "new order",
        ],
        "Order Received 📦": [
            "order received",
            "we received your order",
            "got your order",
            "order confirmed",
        ],
        "Order Shipped 🚚": ["shipped", "dispatched", "on its way", "tracking number"],
        "Order Delivered ✅": [
            "delivered",
            "received the order",
            "got it",
            "delivery done",
        ],
        "Inquiry ❓": [
            "inquiry",
            "question",
            "ask about",
            "details",
            "how much",
            "price",
            "do you have",
        ],
    }

    order_events = []
    for index, row in df.iterrows():
        message = row["message"].lower()
        for event_type, keywords in order_keywords.items():
            if any(keyword in message for keyword in keywords):
                order_events.append(
                    {
                        "datetime": row["datetime"],
                        "user": row["user"],
                        "message": row["message"],
                        "event_type": event_type,
                    }
                )
                break  # Classify message into first matching category

    if not order_events:
        return "No specific order-related phrases found in the chat."

    events_df = pd.DataFrame(order_events).sort_values(by="datetime")

    st.write("### Order-Related Events Timeline:")
    for index, row in events_df.iterrows():
        st.markdown(
            f"**{row['datetime'].strftime('%Y-%m-%d %H:%M')} - {row['user']}**: {row['event_type']} - \"{row['message'][:100]}...\""
        )

    # Basic summary chart
    event_counts = events_df["event_type"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=event_counts.index, y=event_counts.values, palette="pastel", ax=ax)
    ax.set_title("Distribution of Order-Related Events")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return "Order tracking insights:", fig


# 3. Message Classification (Business specific)
def business_message_classification(df):
    if df.empty:
        return "No messages to classify.", None

    message_types_business = {
        "Inquiry 📥": [
            "inquiry",
            "question",
            "ask about",
            "details",
            "how much",
            "price",
            "do you have",
            "can you tell me",
        ],
        "Order 📦": [
            "order placed",
            "i want to buy",
            "confirm order",
            "book",
            "purchase",
            "get me",
            "need to order",
        ],
        "Complaint 🛠️": [
            "complaint",
            "problem",
            "issue",
            "not working",
            "bad service",
            "unhappy",
            "late",
            "wrong item",
            "damaged",
        ],
        "Confirmation ✅": [
            "ok",
            "okay",
            "confirmed",
            "yes",
            "alright",
            "received",
            "thank you",
            "understood",
        ],
        "Payment 💰": [
            "paid",
            "payment",
            "transferred",
            "money",
            "invoice",
            "bill",
            "receipt",
            "amount",
        ],
        "Support/Help ❓": ["help", "support", "assist", "trouble", "fix", "guide me"],
        "Feedback 👍": [
            "feedback",
            "review",
            "suggestion",
            "good service",
            "happy with",
            "excellent",
        ],
    }

    classified_messages = {k: 0 for k in message_types_business.keys()}

    for message in df["message"].dropna().str.lower():
        assigned = False
        for msg_type, keywords in message_types_business.items():
            if any(keyword in message for keyword in keywords):
                classified_messages[msg_type] += 1
                assigned = True
                break
        # If a message doesn't fit any specific category, it's general chat
        # For business context, we might want to count 'General' too, but for simplicity, only specific types are counted.

    total_classified = sum(classified_messages.values())
    if total_classified == 0:
        return "No messages matched defined business categories.", None

    filtered_counts = {k: v for k, v in classified_messages.items() if v > 0}

    labels = filtered_counts.keys()
    sizes = filtered_counts.values()

    fig, ax = plt.subplots()
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "black"},
    )
    ax.axis("equal")
    ax.set_title("Distribution of Business Message Types")
    return "Business message type distribution:", fig


# 4. Deal Conversion Funnel (Conceptual/Keyword-based)
def deal_conversion_funnel(df):
    if df.empty:
        return "No chat data to analyze deal conversion."

    # Define keywords for each stage
    lead_keywords = [
        "interested",
        "looking for",
        "tell me about",
        "want to know",
        "pricing",
    ]
    inquiry_keywords = ["question", "details", "how much", "can you send", "more info"]
    deal_keywords = ["offer", "discount", "deal", "proposal", "quote", "negotiate"]
    payment_keywords = [
        "paid",
        "payment",
        "invoice",
        "bank transfer",
        "upi",
        "transaction",
    ]
    closed_keywords = [
        "confirmed",
        "done",
        "completed",
        "thank you for your order",
        "great service",
    ]

    # Track unique conversations that reach each stage
    # For simplicity, we'll assume each message is a separate "conversation" for funnel purposes,
    # or you'd need to group messages into conversations first.
    # Here, we'll just count messages matching keywords.

    # A more robust funnel would track a single conversation's progression.
    # For this simplified version, we'll count messages that contain keywords for each stage.

    num_leads = (
        df["message"]
        .str.lower()
        .apply(lambda x: any(kw in x for kw in lead_keywords))
        .sum()
    )
    num_inquiries = (
        df["message"]
        .str.lower()
        .apply(lambda x: any(kw in x for kw in inquiry_keywords))
        .sum()
    )
    num_deals = (
        df["message"]
        .str.lower()
        .apply(lambda x: any(kw in x for kw in deal_keywords))
        .sum()
    )
    num_payments = (
        df["message"]
        .str.lower()
        .apply(lambda x: any(kw in x for kw in payment_keywords))
        .sum()
    )
    num_closed = (
        df["message"]
        .str.lower()
        .apply(lambda x: any(kw in x for kw in closed_keywords))
        .sum()
    )

    # Create a funnel-like representation
    funnel_stages = {
        "Leads": num_leads,
        "Inquiries": num_inquiries,
        "Deals": num_deals,
        "Payments": num_payments,
        "Closed": num_closed,
    }

    st.subheader("Conceptual Deal Conversion Funnel (based on keywords)")
    st.write(
        "This funnel tracks the count of messages containing keywords related to each stage."
    )

    # Display as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=list(funnel_stages.keys()),
        y=list(funnel_stages.values()),
        palette="coolwarm",
        ax=ax,
    )
    ax.set_title("Deal Conversion Funnel Stages")
    ax.set_ylabel("Number of Messages (Keyword Matches)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    return "Deal conversion insights:", fig


# 5. Total Payments Extracted (Enhanced)
def extract_total_paid_enhanced(df):
    # Regex to capture numbers with optional currency symbols and various payment-related words
    # It tries to be broad to catch amounts followed by or preceded by payment terms.
    # This might capture some false positives, but it's a starting point.

    # Pattern for "RupeesNo paid", "1000 paid", "₹1000 paid", "Rs. 1000 paid"
    # Pattern for "₹xxxx received", "Paid ₹xxxx", "Payment of xxxx done"

    # This regex attempts to capture numbers that are associated with payment/receipt actions.
    # It looks for optional currency symbols (₹, $, £, Rs., INR), then a number,
    # followed by words like 'paid', 'pays', 'received', 'payment', 'done', 'transfer', 'sent'.
    # Or, it looks for these words followed by a number.

    # Let's refine the regex to be more specific to the user's examples:
    # "RupeesNo paid" -> r'(?:Rupees|₹|Rs\.?|INR)\s*(\d+(?:[.,]\d+)?)\s*paid'
    # "1000 paid" -> r'(\d+(?:[.,]\d+)?)\s*paid'
    # "₹xxxx received" -> r'(?:₹|Rs\.?|INR)\s*(\d+(?:[.,]\d+)?)\s*received'
    # "Paid ₹xxxx" -> r'paid\s*(?:₹|Rs\.?|INR)\s*(\d+(?:[.,]\d+)?)'
    # "Payment of xxxx done" -> r'payment of\s*(\d+(?:[.,]\d+)?)\s*done'

    # Combining these patterns for a more robust extraction
    patterns = [
        r"(?:Rupees|₹|Rs\.?|INR)\s*(\d+(?:[.,]\d+)?)\s*(?:paid|pays|received|done)",  # e.g., "Rupees 1000 paid", "₹1000 received"
        r"(\d+(?:[.,]\d+)?)\s*(?:paid|pays|received|done)",  # e.g., "1000 paid"
        r"(?:paid|pays|received)\s*(?:₹|Rs\.?|INR)\s*(\d+(?:[.,]\d+)?)",  # e.g., "paid ₹1000"
        r"payment of\s*(\d+(?:[.,]\d+)?)\s*done",  # e.g., "payment of 1000 done"
    ]

    total_amount = 0.0
    extracted_details = []

    for index, row in df.iterrows():
        message_lower = row["message"].lower()
        for pattern in patterns:
            matches = re.findall(pattern, message_lower)
            for match in matches:
                try:
                    amount = float(
                        match.replace(",", "")
                    )  # Remove commas for conversion
                    total_amount += amount
                    extracted_details.append(
                        {
                            "datetime": row["datetime"],
                            "user": row["user"],
                            "message": row["message"],
                            "extracted_amount": amount,
                        }
                    )
                except ValueError:
                    # Skip if conversion to float fails
                    continue

    return total_amount, extracted_details


# 6. Client Sentiment Breakdown (using existing sentiment_trend)
# This will be displayed in the UI by calling sentiment_trend and then highlighting specific days.


# 7. Top 5 Repeated Issues
def top_repeated_issues(df, top_k=5):
    if df.empty:
        return "No chat data to identify repeated issues."

    issue_keywords = [
        "late",
        "cancelled",
        "not working",
        "problem",
        "issue",
        "broken",
        "defect",
        "delay",
        "wrong",
        "missing",
        "unhappy",
        "bad",
        "poor",
        "stuck",
        "failed",
    ]

    issue_counts = Counter()

    for message in df["message"].dropna().str.lower():
        for keyword in issue_keywords:
            if keyword in message:
                issue_counts[keyword] += 1

    if not issue_counts:
        return "No common issue keywords detected in the chat."

    top_issues = issue_counts.most_common(top_k)

    issue_df = pd.DataFrame(top_issues, columns=["Issue Keyword", "Frequency"])

    fig, ax = plt.subplots()
    sns.barplot(
        x="Issue Keyword", y="Frequency", data=issue_df, palette="Reds_d", ax=ax
    )
    ax.set_title(f"Top {top_k} Repeated Issue Keywords")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    return "Top repeated issues:", fig


# 8. Working Hour Violation Detector
def working_hour_violation_detector(df, start_hour=10, end_hour=19):  # 10 AM to 7 PM
    if df.empty:
        return "No chat data to detect working hour violations."

    violations = []

    for index, row in df.iterrows():
        msg_hour = row["datetime"].hour

        # Check for messages outside working hours
        if not (start_hour <= msg_hour < end_hour):
            violations.append(
                {
                    "datetime": row["datetime"],
                    "user": row["user"],
                    "message": row["message"],
                    "hour": msg_hour,
                }
            )

    if not violations:
        return f"No messages detected outside working hours ({start_hour}:00 - {end_hour}:00)."

    violations_df = pd.DataFrame(violations)

    # Count violations per user
    user_violations = violations_df["user"].value_counts()

    result_str = f"Detected {len(violations_df)} messages outside typical working hours ({start_hour}:00 - {end_hour}:00):\n"
    for user, count in user_violations.items():
        result_str += f"- **{user}**: {count} messages\n"

    st.write(result_str)
    st.write("### Sample of Messages Outside Working Hours:")
    st.dataframe(
        violations_df[["datetime", "user", "message", "hour"]].head(10)
    )  # Show top 10

    return "Working hour violations detected."


# 9. Follow-Up Needed Flag
def follow_up_needed_flag(df):
    if df.empty:
        return "No chat data to flag follow-ups."

    follow_up_keywords = [
        "still waiting",
        "any update",
        "hello?",
        "where is",
        "when will",
        "chasing",
        "response",
        "reply",
        "urgently",
        "pending",
    ]

    flagged_messages = []

    for index, row in df.iterrows():
        message_lower = row["message"].lower()
        if any(keyword in message_lower for keyword in follow_up_keywords):
            flagged_messages.append(
                {
                    "datetime": row["datetime"],
                    "user": row["user"],
                    "message": row["message"],
                }
            )

    if not flagged_messages:
        return "No messages flagged as needing follow-up based on keywords."

    flagged_df = pd.DataFrame(flagged_messages).sort_values(
        by="datetime", ascending=False
    )

    st.write(
        f"**{len(flagged_df)} messages flagged as potentially needing follow-up:**"
    )
    st.dataframe(flagged_df[["datetime", "user", "message"]])

    return "Messages flagged for follow-up."


# 10. AI Auto-Responder Trainer (Conceptual)
def ai_auto_responder_trainer(df):
    st.write("This feature is conceptual and would involve training a custom AI model.")
    st.markdown(
        """
    To implement an **AI Auto-Responder Trainer**, you would typically:
    1.  **Extract FAQs:** Identify common questions and their answers from your chat history.
        * *Method:* Use clustering on message embeddings, or keyword extraction combined with AI summarization to find question-answer pairs.
    2.  **Generate Template Answers:** Based on identified FAQs, generate concise and helpful template responses.
        * *Method:* Use a large language model (like Gemini) to draft answers for common questions.
    3.  **Train a Custom Chatbot:** Fine-tune a smaller language model or a retrieval-based chatbot on your specific chat data to learn your business's tone and common responses.
        * *Method:* Requires a dataset of (question, answer) pairs and a suitable ML framework (e.g., TensorFlow, PyTorch).
    
    **Current Status:** This is beyond the scope of a real-time Streamlit app without a dedicated backend and significant computational resources for model training. However, the chat analysis (like common phrases, topics) provides foundational data for building such a system.
    """
    )
    st.info(
        "The insights from 'Top 5 Repeated Issues' and 'Identify Key Topics' (under Advanced AI Chat Features) can serve as a starting point for identifying FAQs."
    )


# 11. Smart Summary Generator (AI-based)
def smart_summary_generator(df):
    if df.empty:
        return "No chat data to generate a smart summary."

    # Gather key metrics
    total_messages = len(df)
    unique_users = df["user"].nunique()

    # Calculate sentiment (if not already done)
    if "polarity" not in df.columns:
        df["polarity"] = df["message"].apply(
            lambda msg: TextBlob(msg).sentiment.polarity
        )
    avg_sentiment = df["polarity"].mean()

    # Get response time (overall)
    avg_res_str, _ = calculate_response_times(df)
    if avg_res_str is None:
        avg_res_str = "N/A"

    # Get message classification counts
    message_types_business = {
        "Inquiry": ["inquiry", "question"],
        "Order": ["order", "buy"],
        "Complaint": ["complaint", "problem"],
        "Confirmation": ["confirmed", "ok"],
        "Payment": ["paid", "payment"],
    }
    classified_counts = {k: 0 for k in message_types_business.keys()}
    for message in df["message"].dropna().str.lower():
        for msg_type, keywords in message_types_business.items():
            if any(keyword in message for keyword in keywords):
                classified_counts[msg_type] += 1
                break

    # Get total payments
    total_payments, _ = extract_total_paid_enhanced(df)

    # Construct prompt for AI summary
    prompt_data = f"""
    Generate a concise business summary of the following WhatsApp chat data.
    Focus on key operational metrics and insights.

    Data points:
    - Total messages: {total_messages}
    - Unique participants: {unique_users}
    - Average chat sentiment (polarity): {avg_sentiment:.2f}
    - Overall average response time: {avg_res_str}
    - Message type counts: {classified_counts}
    - Total payments extracted: ₹{total_payments:,.2f}

    Format: "In [Month/Period], you had [X] chats with [Y] clients. [Z] orders were closed, [A] complaints raised. Avg. response time: [B]."
    Adjust the summary based on available data and make it sound professional and insightful.
    """

    # Add sample chat for context if not too long
    chat_sample = " ".join(df["message"].dropna().tolist())
    if len(chat_sample) > 2000:
        chat_sample = chat_sample[:2000] + "..."
    prompt_data += f"\n\nSample chat content: {chat_sample}"

    try:
        response = model.generate_content(prompt_data)
        return response.text
    except Exception as e:
        return f"Error generating smart summary: {e}"


# ===================== Streamlit App =====================
st.set_page_config(layout="wide")
st.title("🔐 WhatsApp Chat Analyzer - Login Required")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:  # Initialize page state
    st.session_state.page = "main_app"  # Default to main analysis page
if "media_messages_count" not in st.session_state:
    st.session_state["media_messages_count"] = 0


if choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.session_state.username = (
                username  # Store username for personalized greetings
            )
            st.success(f"Welcome {username}!")
            st.rerun()  # Rerun to hide login form and show main content
        else:
            st.error("Invalid credentials")

elif choice == "Register":
    st.subheader("Create New Account")
    new_user = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Register"):
        if register_user(new_user, new_password):
            st.success("Registration successful. Please log in.")
        else:
            st.error("Username already exists.")

if not st.session_state.authenticated:
    st.stop()  # Stop execution if not authenticated

# --- Main Application Logic (after authentication) ---

# Define uploaded_file and df at a higher scope
uploaded_file = st.file_uploader("📂 Upload your WhatsApp chat (.txt file)", type="txt")
df = pd.DataFrame()  # Initialize an empty DataFrame

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")
    df = preprocess_chat(
        raw_text
    )  # preprocess_chat now updates st.session_state['media_messages_count']
    if df.empty and st.session_state["media_messages_count"] == 0:
        st.warning(
            "⚠️ No valid messages found in the uploaded file after preprocessing (only media or empty)."
        )
        uploaded_file = None  # Reset uploaded_file if no valid data
    elif df.empty and st.session_state["media_messages_count"] > 0:
        st.warning(
            f"⚠️ Only media messages ({st.session_state['media_messages_count']} found) in the uploaded file. No text messages to analyze."
        )
        # We can still proceed with some stats if only media messages were there, but most analysis will be empty
        df = pd.DataFrame(
            columns=["datetime", "user", "message", "polarity"]
        )  # Ensure df is not truly empty for further checks

# Navigation for pages
st.sidebar.markdown("---")
if st.sidebar.button("📊 Chat Analysis Dashboard"):
    st.session_state.page = "main_app"
if st.sidebar.button("💬 Launch Smart WhatsApp Tools"):
    st.session_state.page = "smart_tools"
if st.sidebar.button("❤️ Relationship Features"):  # NEW MENU ITEM
    st.session_state.page = "relationship_features"
if st.sidebar.button("💼 Business Features"):  # NEW BUSINESS MENU ITEM
    st.session_state.page = "business_features"


# Render pages based on session state

   
if (
    st.session_state.page == "main_app"
):  # This block now handles the main analysis dashboard
    st.title("📊 Chat Analysis Dashboard")
    st.write("Upload your WhatsApp chat file to get insights.")

    if df.empty and st.session_state["media_messages_count"] == 0:
        st.warning("⚠️ Please upload a WhatsApp chat file above to start the analysis.")
    else:
        st.sidebar.header("🧮 Choose Analysis")
        all_users = df["user"].unique().tolist()
        selected_users = st.sidebar.multiselect("👤 Filter by User(s)", all_users)
        start_date = st.sidebar.date_input(
            "📅 Start Date",
            value=(
                df["datetime"].min().date() if not df.empty else datetime.now().date()
            ),
        )
        end_date = st.sidebar.date_input(
            "📅 End Date",
            value=(
                df["datetime"].max().date() if not df.empty else datetime.now().date()
            ),
        )
        filtered_df = filter_data(
            df, selected_users, start_date, end_date
        )  # Use a new variable for filtered df

        st.header("Overall Chat Statistics")
        total_text_messages = len(filtered_df)
        total_messages_incl_media = (
            total_text_messages + st.session_state["media_messages_count"]
        )

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Messages (Text+Media)", total_messages_incl_media)
        with col_stat2:
            st.metric("Text Messages Analyzed", total_text_messages)
        with col_stat3:
            st.metric(
                "Unique Participants",
                filtered_df["user"].nunique() if not filtered_df.empty else 0,
            )
        with col_stat4:
            st.metric(
                "Avg. Msgs/Day (Text Only)",
                (
                    f"{filtered_df.groupby(filtered_df['datetime'].dt.date).size().mean():.2f}"
                    if not filtered_df.empty
                    else "N/A"
                ),
            )

        if st.sidebar.checkbox("📊 Message Type Distribution"):
            st.subheader("📄 Message Type Distribution (Text vs. Media)")
            fig_type, msg_type_error = message_type_distribution(
                total_text_messages, st.session_state["media_messages_count"]
            )
            if fig_type:
                st.pyplot(fig_type)
            else:
                st.info(msg_type_error)

        if filtered_df.empty:
            st.warning(
                "No text messages found for the selected filters. Some analyses may not be available."
            )
        else:
            if st.sidebar.checkbox("📄 Show Raw Data"):
                st.subheader("Raw Chat Data (Filtered Text Messages)")
                st.dataframe(filtered_df)

            if st.sidebar.checkbox("📊 User-Specific Stats"):
                st.subheader("📈 User Message Stats")
                avg_lengths, avg_msgs_per_day = user_message_stats(filtered_df)
                st.write("**📏 Avg. Message Length by User:**")
                st.dataframe(avg_lengths.rename("Avg Length"))
                st.write("**📆 Avg. Messages per Day by User:**")
                st.dataframe(avg_msgs_per_day.rename("Avg Msgs/Day"))

            if st.sidebar.checkbox("🎭 Sentiment Analysis"):
                st.subheader("🎯 Sentiment Analysis by User")
                # Ensure 'polarity' is calculated for filtered_df if sentiment analysis is selected
                filtered_df["polarity"] = filtered_df["message"].apply(
                    lambda msg: TextBlob(msg).sentiment.polarity
                )
                sentiments = sentiment_analysis(filtered_df)
                st.write("**User Sentiment Polarity (−1 = Negative, +1 = Positive):**")
                st.dataframe(sentiments.rename("Polarity"))
                most_pos = sentiments.idxmax()
                most_neg = sentiments.idxmin()

                st.success(
                    f"😊 Most Positive User: {most_pos} ({sentiments.max():.2f})"
                )
                st.error(f"😠 Most Negative User: {most_neg} ({sentiments.min():.2f})")

                st.subheader(
                    f"🔍 Messages Contributing to Negative Sentiment for {most_neg}"
                )
                neg_msgs = filtered_df[
                    (filtered_df["user"] == most_neg) & (filtered_df["polarity"] < 0)
                ].sort_values(by="polarity")
                if neg_msgs.empty:
                    st.info("No clearly negative messages were found for this user.")
                else:
                    st.markdown(
                        f"These are the messages from **{most_neg}** with negative sentiment scores, which contributed to their overall polarity of **{sentiments[most_neg]:.2f}**."
                    )
                    st.dataframe(
                        neg_msgs[["datetime", "message", "polarity"]].reset_index(
                            drop=True
                        )
                    )

                st.subheader(f"✨ Positive Messages from {most_pos}")
                pos_msgs = filtered_df[
                    (filtered_df["user"] == most_pos) & (filtered_df["polarity"] > 0)
                ].sort_values(by="polarity", ascending=False)
                if pos_msgs.empty:
                    st.info("No clearly positive messages were found for this user.")
                else:
                    st.dataframe(
                        pos_msgs[["datetime", "message", "polarity"]].reset_index(
                            drop=True
                        )
                    )

                # New: Sentiment Trend over Time
                st.subheader("📈 Sentiment Trend Over Time")
                sentiment_trend_data = sentiment_trend(filtered_df)
                if not sentiment_trend_data.empty:
                    fig_sentiment, ax_sentiment = plt.subplots(figsize=(10, 5))
                    sns.lineplot(
                        x="date",
                        y="polarity",
                        data=sentiment_trend_data,
                        ax=ax_sentiment,
                    )
                    ax_sentiment.set_title("Average Daily Sentiment Polarity")
                    ax_sentiment.set_xlabel("Date")
                    ax_sentiment.set_ylabel("Average Polarity")
                    ax_sentiment.axhline(0, color="grey", linestyle="--", linewidth=0.8)
                    st.pyplot(fig_sentiment)
                else:
                    st.info("Not enough data to plot sentiment trend.")

            if st.sidebar.checkbox("📊 Multi-User Comparison"):
                st.subheader("🔄 Multi-User Message Comparison")
                user_counts = filtered_df["user"].value_counts()
                fig, ax = plt.subplots()
                sns.barplot(
                    x=user_counts.index, y=user_counts.values, ax=ax, palette="coolwarm"
                )
                ax.set_title("Messages per User")
                ax.set_ylabel("Messages")
                ax.set_xticklabels(user_counts.index, rotation=45, ha="right")
                st.pyplot(fig)

            if st.sidebar.checkbox("📅 Weekly Activity"):
                st.subheader("🗓️ Weekly Chat Activity")
                weekly = weekly_activity(filtered_df)
                fig, ax = plt.subplots()
                sns.barplot(x=weekly.index, y=weekly.values, palette="viridis", ax=ax)
                ax.set_ylabel("Message Count")
                ax.set_xlabel("Day")
                st.pyplot(fig)

            if st.sidebar.checkbox("📆 Monthly Activity"):
                st.subheader("📈 Monthly Chat Activity")
                monthly = monthly_activity(filtered_df)
                fig, ax = plt.subplots()
                monthly.plot(kind="bar", color="skyblue", ax=ax)
                ax.set_ylabel("Messages")
                ax.set_xlabel("Month")
                st.pyplot(fig)

            if st.sidebar.checkbox("🕒 Hourly Activity"):
                st.subheader("⌛ Most Active Hours")
                hourly = hourly_distribution(filtered_df)
                fig, ax = plt.subplots()
                sns.lineplot(x=hourly.index, y=hourly.values, marker="o", ax=ax)
                ax.set_xticks(range(0, 24))
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Message Count")
                st.pyplot(fig)

            # New: Top Active Hours List
            if st.sidebar.checkbox("🔝 Top Active Hours (List)"):
                st.subheader("⏱️ Top 5 Most Active Hours (by message count)")
                top_hours = top_active_hours_list(filtered_df, top_k=5)
                if isinstance(top_hours, str):
                    st.info(top_hours)
                else:
                    st.dataframe(top_hours.rename("Message Count"))

            if st.sidebar.checkbox("📊 Activity Heatmap"):
                st.subheader("🔥 Activity Heatmap (Messages per Hour by Day)")
                pivot = heatmap_data(filtered_df)
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(pivot, cmap="YlGnBu", ax=ax, linewidths=0.5)
                ax.set_title("Messages per Hour by Day")
                st.pyplot(fig)

            # New: Most Active Hours by Day of Week
            if st.sidebar.checkbox("📊 Active Hours by Day of Week"):
                st.subheader("🔥 Most Active Hours by Day of Week")
                active_hours_data = active_hours_by_day(filtered_df)
                fig_active_hours, ax_active_hours = plt.subplots(figsize=(12, 7))
                sns.heatmap(
                    active_hours_data,
                    cmap="YlGnBu",
                    ax=ax_active_hours,
                    linewidths=0.5,
                    fmt="g",
                )
                ax_active_hours.set_title("Message Count Heatmap (Day vs. Hour)")
                ax_active_hours.set_xlabel("Hour of Day")
                ax_active_hours.set_ylabel("Day of Week")
                st.pyplot(fig_active_hours)

            if st.sidebar.checkbox("🗣️ Most Common Words"):
                st.subheader("💬 Most Common Words")
                common = common_words(filtered_df)
                common_df = pd.DataFrame(common, columns=["Word", "Frequency"])
                st.dataframe(common_df)

            # New: Most Common Phrases
            if st.sidebar.checkbox("🗣️ Most Common Phrases"):
                st.subheader("💬 Most Common Phrases (N-word)")
                num_words_phrase = st.slider(
                    "Phrase Length (N-gram)",
                    min_value=2,
                    max_value=4,
                    value=2,
                    key="phrase_length_slider",
                )
                top_k_phrases = st.slider(
                    "Top K Phrases",
                    min_value=5,
                    max_value=20,
                    value=10,
                    key="top_k_phrases_slider",
                )

                common_phr = common_phrases(
                    filtered_df, num_words_phrase, top_k_phrases
                )
                if common_phr:
                    phrases_df = pd.DataFrame(
                        [(" ".join(phrase), count) for phrase, count in common_phr],
                        columns=["Phrase", "Frequency"],
                    )
                    st.dataframe(phrases_df)
                else:
                    st.info(
                        "Not enough data to find common phrases or no phrases found."
                    )

            if st.sidebar.checkbox("☁️ Word Cloud"):
                st.subheader("☁️ Word Cloud")
                wc = generate_wordcloud(filtered_df)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

            if st.sidebar.checkbox("🙋 Chat Frequency by User"):
                st.subheader("📊 Chat Frequency by User")
                freq = user_frequency(filtered_df)
                fig, ax = plt.subplots()
                sns.barplot(x=freq.index, y=freq.values, palette="Set2", ax=ax)
                ax.set_xticklabels(freq.index, rotation=45, ha="right")
                ax.set_ylabel("Messages")
                st.pyplot(fig)

            if st.sidebar.checkbox("😂 Emoji Usage"):
                st.subheader("😂 Top Emojis Used")
                emojis = emoji_counter(filtered_df)
                emoji_df = pd.DataFrame(emojis, columns=["Emoji", "Count"])
                st.dataframe(emoji_df)

            if st.sidebar.checkbox("🔮 Predict Busiest Month"):
                st.subheader("📅 Predicted Busiest Upcoming Month")
                month, forecast, reason = predict_busiest_month(filtered_df)
                if month:
                    st.success(f"🚀 Predicted busiest month is: **{month}**")
                    st.write(reason)
                    st.line_chart(forecast)
                else:
                    st.warning(reason)

            if st.sidebar.checkbox("💬 Predict Future Messages"):
                st.subheader("🗯️ AI-Predicted Future Messages")
                predictions = generate_future_messages(filtered_df)
                for msg in predictions:
                    st.write(f"• {msg}")

            if st.sidebar.checkbox("💡 Predict Future Words"):
                st.subheader("🔤 Predicted Future Words")
                predicted_words = predict_future_words(filtered_df)
                st.write(" ".join(predicted_words))

            if st.sidebar.checkbox("🗓️ Predict Most Active Days"):
                st.subheader("🗓️ Likely Active Weekdays")
                days, reason = predict_active_days(filtered_df)
                st.info(reason)

            if st.sidebar.checkbox("⏳ Predict Most Active Hours"):
                st.subheader("⏰ Likely Active Hours")
                hours, reason = predict_active_hours(filtered_df)
                st.info(reason)

            if st.sidebar.checkbox("📈 Message Volume Trend"):
                st.subheader("📉 Message Trend Analysis")
                st.info(detect_message_trend(filtered_df))


elif st.session_state.page == "relationship_features":  # RELATIONSHIP FEATURES PAGE
    st.title("❤️ Relationship Features")
    st.write(
        "Analyze your chat with your partner to gain insights into your relationship dynamics."
    )
    st.warning(
        "⚠️ **Disclaimer:** These analyses are generated by AI and heuristics based on textual patterns in your chat. They are for entertainment and insight purposes only and cannot fully capture the complexities of human relationships. Do not use them as the sole indicator of your relationship's health."
    )

    if df.empty:
        st.warning(
            "Please upload a WhatsApp chat file on the 'Chat Analysis Dashboard' page to enable these features."
        )
    elif df["user"].nunique() != 2:
        st.warning(
            "This section is designed for chats between exactly two participants. Please upload a two-person chat export for accurate analysis."
        )
    else:
        # Pre-calculate polarity for all relationship features
        df["polarity"] = df["message"].apply(
            lambda msg: TextBlob(msg).sentiment.polarity
        )

        st.markdown("---")
        st.markdown("### ❤️ Love Score & Overall Dynamics (AI)")

        if st.button("Calculate Love Score & Dynamics"):
            with st.spinner(
                "Analyzing your chat for love score and dynamics... This may take a moment."
            ):
                # Love Score
                status, score, explanation, recommendations = calculate_love_score(
                    df
                )  # Use df directly, as filtering is not typical for this feature
                if status == "Success":
                    if score is not None:
                        st.subheader(f"Your Relationship Love Score: {score}/100 ❤️")
                        st.markdown("---")
                        st.subheader("Detailed Explanation:")
                        st.write(explanation)
                        st.markdown("---")
                        st.subheader("Recommendations:")
                        st.write(recommendations)
                    else:
                        st.error(
                            "Could not determine a love score from the AI analysis. Please try again with a longer or more active chat."
                        )
                else:
                    st.error(status)

        st.markdown("---")
        if st.checkbox("📈 Relationship Timeline Insights"):
            st.subheader("Relationship Timeline Insights")
            first_msg_date, total_days_chatted, longest_gap, longest_streak = (
                calculate_relationship_timeline_insights(df)
            )
            if first_msg_date:
                st.write(f"**First Message Date:** {first_msg_date}")
                st.write(f"**Total Days Chatting:** {total_days_chatted} days")
                st.write(
                    f"**Longest Gap Between Replies (between different users):** {longest_gap}"
                )
                st.write(
                    f"**Longest Consistent Daily Chatting Streak:** {longest_streak} days"
                )
            else:
                st.info("Not enough data to generate relationship timeline insights.")

        st.markdown("---")
        if st.checkbox("💬 Conversation Balance"):
            st.subheader("Conversation Balance")
            (
                dominance_text,
                longer_msgs_user,
                longer_msgs_avg,
                initiator_text,
                user_counts,
            ) = get_conversation_balance(df)
            if dominance_text and user_counts is not None and len(user_counts) == 2:
                st.write(dominance_text)
                st.write(
                    f"**User who sends longer messages on average:** **{longer_msgs_user}** (Avg. length: {longer_msgs_avg:.2f} chars)"
                )
                st.write(initiator_text)

                fig_balance, ax_balance = plt.subplots()
                ax_balance.pie(
                    user_counts.values,
                    labels=user_counts.index,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=["#ff9999", "#66b3ff"],
                )
                ax_balance.axis(
                    "equal"
                )  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig_balance)

            else:
                st.info(dominance_text)  # This will output the error message

        st.markdown("---")
        if st.checkbox("💖 Top 5 Most Positive Messages"):
            st.subheader("Top 5 Most Positive Messages")
            romantic_msgs = top_romantic_messages(df)
            if not romantic_msgs.empty:
                for index, row in romantic_msgs.iterrows():
                    st.markdown(
                        f"**{row['user']}** on {row['datetime'].strftime('%Y-%m-%d %H:%M')}:"
                    )
                    st.info(f"\"{row['message']}\" (Polarity: {row['polarity']:.2f})")
            else:
                st.info("No significantly positive messages found to highlight.")

        st.markdown("---")
        if st.checkbox("⏳ Response Time Patterns (Per User)"):
            st.subheader("Response Time Patterns (Per User)")
            avg_res_readable, avg_res_sec, fastest_replier = (
                calculate_response_times_per_user(df)
            )
            if isinstance(avg_res_readable, str):
                st.info(avg_res_readable)
            else:
                st.write("**Average Response Times:**")
                for user, time_str in avg_res_readable.items():
                    st.write(f"- **{user}:** {time_str}")
                if fastest_replier:
                    st.success(f"**Fastest Replier:** {fastest_replier}")
                else:
                    st.info("Could not determine the fastest replier.")

        st.markdown("---")
        if st.checkbox("🧠 Personality Traits (AI-based Guess)"):
            st.subheader("Personality Traits (AI-based Guess)")
            personality_output = guess_personality_traits(df)
            st.write(personality_output)

        st.markdown("---")
        if st.checkbox("📌 Conflict & Recovery Patterns"):
            st.subheader("Conflict & Recovery Patterns")
            conflict_recovery_status = detect_conflict_and_recovery(df)
            st.write(conflict_recovery_status)

        st.markdown("---")
        if st.checkbox("🧭 Compatibility Radar Chart"):
            st.subheader("Compatibility Radar Chart")
            radar_fig, radar_error = create_compatibility_radar_chart(df)
            if radar_fig:
                st.pyplot(radar_fig)
            else:
                st.info(radar_error)

        st.markdown("---")
        if st.checkbox("💞 Emotional Compatibility Meter"):
            st.subheader("Emotional Compatibility Meter")
            compat_result = emotional_compatibility_meter(df)
            if isinstance(compat_result, str):
                st.info(compat_result)
            else:
                score, interpretation, avg1, avg2, std1, std2 = compat_result
                st.metric("Emotional Compatibility Score", f"{score:.1f}/100")
                st.write(f"**Interpretation:** {interpretation}")
                st.write(
                    f"- User 1 Average Sentiment: {avg1:.2f} (Emotional Range: {std1:.2f})"
                )
                st.write(
                    f"- User 2 Average Sentiment: {avg2:.2f} (Emotional Range: {std2:.2f})"
                )

        st.markdown("---")
        if st.checkbox("💬 Message Type Classifier"):
            st.subheader("Message Type Classifier")
            message_type_output, msg_type_fig = classify_message_types(df)
            if msg_type_fig:
                st.pyplot(msg_type_fig)
                st.write(message_type_output)
            else:
                st.info(message_type_output)

        st.markdown("---")
        if st.checkbox("📖 Relationship Diary Generator (AI)"):
            st.subheader("Relationship Diary Entry (Powered by Gemini)")
            if st.button("Generate Diary Entry"):
                with st.spinner("Generating diary entry..."):
                    diary_entry = generate_relationship_diary(df)
                    st.write(diary_entry)
            else:
                st.info(
                    "Click the button to generate a summary of your relationship's chat history."
                )

        st.markdown("---")
        if st.checkbox("📈 Mood Shifts Over Time"):
            st.subheader("Mood Shifts Over Time (Sentiment Trend)")
            sentiment_trend_data = sentiment_trend(df)
            if not sentiment_trend_data.empty:
                fig_sentiment, ax_sentiment = plt.subplots(figsize=(10, 5))
                sns.lineplot(
                    x="date", y="polarity", data=sentiment_trend_data, ax=ax_sentiment
                )
                ax_sentiment.set_title("Average Daily Sentiment Polarity")
                ax_sentiment.set_xlabel("Date")
                ax_sentiment.set_ylabel("Average Polarity")
                ax_sentiment.axhline(0, color="grey", linestyle="--", linewidth=0.8)
                st.pyplot(fig_sentiment)
            else:
                st.info("Not enough data to plot sentiment trend.")

        st.markdown("---")
        if st.checkbox("💌 Love Language Classifier"):
            st.subheader("Love Language Classifier (Based on Chat Content)")
            love_lang_output, love_lang_fig = classify_love_language(df)
            if love_lang_fig:
                st.pyplot(love_lang_fig)
                st.write(love_lang_output)
            else:
                st.info(love_lang_output)

        st.markdown("---")
        if st.checkbox("📊 Attachment Style Prediction (Experimental)"):
            st.subheader("Attachment Style Prediction (Experimental)")
            attachment_styles = predict_attachment_style(df)
            if isinstance(attachment_styles, str):
                st.info(attachment_styles)
            else:
                for user, data in attachment_styles.items():
                    st.write(f"**{user}:** {data['style']}")
                    st.write(f"*{data['explanation']}*")


elif st.session_state.page == "business_features":  # BUSINESS FEATURES PAGE
    st.title("💼 Business Features")
    st.write("Analyze your business-related WhatsApp chats for operational insights.")
    st.warning(
        "⚠️ **Disclaimer:** These analyses are based on textual patterns and heuristics. For critical business decisions, always cross-reference with official records."
    )

    if df.empty:
        st.warning(
            "Please upload a WhatsApp chat file on the 'Chat Analysis Dashboard' page to enable these features."
        )
    else:
        # Pre-calculate polarity for business features too
        df["polarity"] = df["message"].apply(
            lambda msg: TextBlob(msg).sentiment.polarity
        )

        st.markdown("---")
        st.markdown("### 🔍 Core Analytics")

        if st.checkbox("📊 Response Time Insights"):
            st.subheader("Response Time Insights (Business Context)")
            res_str, res_fig = business_response_time_insights(df)
            if res_fig:
                st.write(res_str)
                st.pyplot(res_fig)
            else:
                st.info(res_str)

        st.markdown("---")
        if st.checkbox("📦 Order Tracker"):
            st.subheader("Order Tracking (Keyword-based)")
            order_output, order_fig = order_tracker(df)
            if order_fig:
                st.write(order_output)
                st.pyplot(order_fig)
            else:
                st.info(order_output)

        st.markdown("---")
        if st.checkbox("💬 Message Classification"):
            st.subheader("Business Message Type Classification")
            biz_msg_output, biz_msg_fig = business_message_classification(df)
            if biz_msg_fig:
                st.pyplot(biz_msg_fig)
                st.write(biz_msg_output)
            else:
                st.info(biz_msg_output)

        st.markdown("---")
        if st.checkbox("🎯 Deal Conversion Funnel"):
            st.subheader("Conceptual Deal Conversion Funnel")
            funnel_output, funnel_fig = deal_conversion_funnel(df)
            if funnel_fig:
                st.write(funnel_output)
                st.pyplot(funnel_fig)
            else:
                st.info(funnel_output)

        st.markdown("---")
        if st.checkbox("💰 Total Payments Extracted"):
            st.subheader("Total Payments Extracted from Chat")
            total_paid_amount, extracted_details = extract_total_paid_enhanced(df)
            if total_paid_amount > 0:
                st.success(f"**Total Amount Detected:** ₹{total_paid_amount:,.2f}")
                st.info(
                    "This includes messages like '1000 paid', '₹1000 received', 'Paid ₹1000', 'Payment of 1000 done'."
                )
                if extracted_details:
                    st.write("### Extracted Payment Details:")
                    payments_df = pd.DataFrame(extracted_details)
                    st.dataframe(
                        payments_df[["datetime", "user", "extracted_amount", "message"]]
                    )
            else:
                st.info(
                    "No payment messages (e.g., '1000 paid', '₹1000 received') found in the chat."
                )

        st.markdown("---")
        if st.checkbox("💡 Client Sentiment Breakdown"):
            st.subheader("Client Sentiment Breakdown")
            st.write("Analyze the sentiment of messages from clients over time.")

            # Assuming the 'business' is one user and 'clients' are others, or just overall sentiment.
            # For simplicity, we'll show overall sentiment trend.
            sentiment_trend_data = sentiment_trend(df)
            if not sentiment_trend_data.empty:
                fig_sentiment, ax_sentiment = plt.subplots(figsize=(10, 5))
                sns.lineplot(
                    x="date", y="polarity", data=sentiment_trend_data, ax=ax_sentiment
                )
                ax_sentiment.set_title(
                    "Average Daily Sentiment Polarity (Overall Chat)"
                )
                ax_sentiment.set_xlabel("Date")
                ax_sentiment.set_ylabel("Average Polarity")
                ax_sentiment.axhline(0, color="grey", linestyle="--", linewidth=0.8)
                st.pyplot(fig_sentiment)

                # Highlight top positive/negative days
                if not sentiment_trend_data.empty:
                    most_positive_day = sentiment_trend_data.loc[
                        sentiment_trend_data["polarity"].idxmax()
                    ]
                    most_negative_day = sentiment_trend_data.loc[
                        sentiment_trend_data["polarity"].idxmin()
                    ]
                    st.success(
                        f"**Most Positive Day:** {most_positive_day['date'].strftime('%Y-%m-%d')} (Polarity: {most_positive_day['polarity']:.2f})"
                    )
                    st.error(
                        f"**Most Negative Day:** {most_negative_day['date'].strftime('%Y-%m-%d')} (Polarity: {most_negative_day['polarity']:.2f})"
                    )
            else:
                st.info("Not enough data to plot client sentiment trend.")

        st.markdown("---")
        if st.checkbox("🗂️ Top 5 Repeated Issues"):
            st.subheader("Top Repeated Issues (Keywords)")
            issues_output, issues_fig = top_repeated_issues(df)
            if issues_fig:
                st.write(issues_output)
                st.pyplot(issues_fig)
            else:
                st.info(issues_output)

        st.markdown("---")
        if st.checkbox("⏱️ Working Hour Violation Detector"):
            st.subheader("Working Hour Violation Detector")
            st.write(
                "Checks for messages sent/received outside defined working hours (10 AM - 7 PM)."
            )
            violation_status = working_hour_violation_detector(df)
            st.write(violation_status)

        st.markdown("---")
        if st.checkbox("📅 Follow-Up Needed Flag"):
            st.subheader("Follow-Up Needed Flag")
            follow_up_status = follow_up_needed_flag(df)
            st.write(follow_up_status)

        st.markdown("---")
        st.markdown("### 🚀 Bonus: Automation & Export")

        if st.checkbox("📝 Smart Summary Generator (AI)"):
            st.subheader("Smart Business Summary (Powered by Gemini)")
            if st.button("Generate Business Summary"):
                with st.spinner("Generating smart business summary..."):
                    business_summary = smart_summary_generator(df)
                    st.write(business_summary)
            else:
                st.info(
                    "Click the button to get an AI-generated summary of your business chat performance."
                )

        if st.checkbox("🧠 AI Auto-Responder Trainer (Conceptual)"):
            st.subheader("AI Auto-Responder Trainer (Conceptual)")
            ai_auto_responder_trainer(df)
