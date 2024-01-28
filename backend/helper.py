import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from backend import preprocessor
from sklearn.preprocessing import OrdinalEncoder
from textblob import TextBlob
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import emoji
import streamlit as st




extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    # Fetch number of media messages (either '<Media omitted>\n' or 'image omitted')
    num_media_messages = df[(df['message'] == '<Media omitted>\n')].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('backend/stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('backend/stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])
        emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def analyze_sentiment(message):
    blob = TextBlob(message)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def message_length_analysis(selected_participant,df):
    filtered_df = df[df['user'] == selected_participant] if selected_participant != 'Overall' else df
    filtered_df['message_length'] = filtered_df['message'].apply(lambda msg: len(msg))
    average_length = filtered_df['message_length'].mean()
    st.write(f"Average Message Length for {selected_participant}: {average_length:.2f}")


# Function for busiest hours analysis
def busiest_hours_analysis(df):
    busiest_hours = df['hour'].value_counts()
    st.bar_chart(busiest_hours)


# Function for message count by month
def message_count_by_month(selected_participant,df):
    filtered_df = df[df['user'] == selected_participant] if selected_participant != 'Overall' else df
    message_count_per_month = filtered_df.groupby(['year', 'month']).count()['message'].reset_index()
    st.dataframe(message_count_per_month)


# Function for top emojis used
def top_emojis_used(selected_participant,df):
    filtered_df = df[df['user'] == selected_participant] if selected_participant != 'Overall' else df
    emojis = [c for message in filtered_df['message'] for c in message if c in emoji.UNICODE_EMOJI['en']]
    top_emojis = Counter(emojis).most_common()
    st.write(f"Top Emojis Used by {selected_participant}: {top_emojis}")


# Function for greeting and farewell analysis
def greeting_farewell_analysis(selected_participant,df):
    filtered_df = df[df['user'] == selected_participant] if selected_participant != 'Overall' else df

    greetings = filtered_df['message'].apply(lambda msg: 'hello' in msg.lower() or 'hi' in msg.lower()).sum()
    farewells = filtered_df['message'].apply(lambda msg: 'goodbye' in msg.lower() or 'bye' in msg.lower()).sum()
    birthdays = filtered_df['message'].apply(
        lambda msg: 'happy birthday' in msg.lower() or 'happiest birthday' in msg.lower()).sum()

    total_messages = filtered_df.shape[0]
    greeting_percentage = (greetings / total_messages) * 100
    farewell_percentage = (farewells / total_messages) * 100
    birthday_percentage = (birthdays / total_messages) * 100

    # Create a pie chart
    labels = ['Greetings', 'Farewells', 'Birthday Wishes']
    sizes = [greeting_percentage, farewell_percentage, birthday_percentage]
    colors = ['yellowgreen', 'lightskyblue', 'lightcoral']
    explode = (0.1, 0, 0)  # explode the first slice

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')

    greetings = filtered_df['message'].apply(lambda msg: 'hello' in msg.lower() or 'hi' in msg.lower()).sum()
    farewells = filtered_df['message'].apply(lambda msg: 'goodbye' in msg.lower() or 'bye' in msg.lower()).sum()
    birthdays = filtered_df['message'].apply(
        lambda msg: 'happy birthday' in msg.lower() or 'happiest birthday' in msg.lower()).sum()

    st.write(f"Total Greetings by {selected_participant}: {greetings}")
    st.write(f"Total Farewells by {selected_participant}: {farewells}")
    st.write(f"Total Birthday Wishes by {selected_participant}: {birthdays}")

    st.pyplot(fig)


# Function for topic analysis using LDA
with open('backend/stop_hinglish.txt', 'r') as f:
    stop_words = set(f.read().splitlines())

# Function for topic analysis using LDA with heuristic topic naming
# Load the stop words from the file
with open('backend/stop_hinglish.txt', 'r') as f:
    stop_words = set(f.read().splitlines())

#Reply

def longest_reply_user(df):
    # Ordinal encoders will encode each user with its own number
    user_encoder = OrdinalEncoder()
    df['User Code'] = user_encoder.fit_transform(df['user'].values.reshape(-1, 1))

    # Find replies
    message_senders = df['User Code'].values
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
    sender_changed[0] = False
    is_reply = sender_changed & ~df['user'].eq('group_notification')

    df['Is Reply'] = is_reply

    # Calculate times based on replies
    reply_times, indices = preprocessor.calculate_times_on_trues(df, 'Is Reply')
    reply_times_df_list = []
    reply_time_index = 0
    for i in range(0, len(df)):
        if i in indices:
            reply_times_df_list.append(reply_times[reply_time_index].astype("timedelta64[m]").astype("float"))
            reply_time_index += 1
        else:
            reply_times_df_list.append(0)

    df['Reply Time'] = reply_times_df_list

    # Calculate the maximum reply time for each user
    max_reply_times = df.groupby('user')['Reply Time'].max()

    # Find the user with the longest reply time
    max_reply_user = max_reply_times.idxmax()
    max_reply_time = max_reply_times.max()

    return max_reply_user, max_reply_time


def _create_wide_area_fig(df : pd.DataFrame, legend : bool = True):
    fig, ax = plt.subplots(figsize=(12,5))
    df.plot(
        alpha=0.6,
        cmap=plt.get_cmap('viridis'),
        ax=ax,
        stacked=True
    )
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    if legend:
        ax.legend(df['user'])
    return fig


def create_narrow_pie_fig(df : pd.DataFrame):
    narrow_figsize = (6, 5)
    cmap = plt.get_cmap('viridis')
    fig1, ax = plt.subplots(figsize=narrow_figsize)
    df.plot(kind='pie', cmap=cmap, ax=ax, autopct='%1.1f%%', explode=[0.015] * len(df.index.unique()))
    centre_circle = plt.Circle((0, 0), 0.80, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    ax.set_ylabel('')
    return fig


def message_count_aggregated_graph(df):
    subject_df = df.groupby('user').count()['message'].sort_values(ascending=False)
    most_messages_winner = subject_df.index[subject_df.argmax()]
    fig = create_narrow_pie_fig(subject_df)
    return fig, most_messages_winner


#This are not working work on this!

# # def conversation_starter_graph( df):
# #     subject_df = df[df['Conv change']].groupby('Subject').count()['Reply time']
# #     fig = create_narrow_pie_fig(subject_df)
# #     most_messages_winner = subject_df.index[subject_df.argmax()]
# #     return fig, most_messages_winner
#
#
#
# def create_messages_per_week_graph(df: pd.DataFrame):
#     # Convert 'Date' column to datetime if it's not already
#     # df['Date'] = pd.to_datetime(df['Date'])
#
#     # Makes the first graph
#     date_df = df.groupby('date')[df['user']].sum().resample('W').sum()
#     fig = _create_wide_area_fig(date_df)
#
#     max_message_count = date_df[df['user']].sum(axis=1).max()
#     max_message_count_date = date_df.index[date_df[df['user']].sum(axis=1).argmax()]
#     return fig, max_message_count, max_message_count_date
#
# def create_average_wpm_graph( df : pd.DataFrame):
#     other_y_columns = [f"{subject}_mlength" for subject in df['user'].unique()]
#     date_avg_df = df[other_y_columns].resample('W').mean()
#     fig = _create_wide_area_fig(date_avg_df)
#     return fig







