import base64
import os
import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
from backend import preprocessor
from backend import helper


#Layout
st.set_page_config(
    page_title="Whatsapp Chat Analyzer",
    layout="wide",
    initial_sidebar_state="expanded")


def display_chat_message(message):
    st.markdown(f'<div class="chat-message">{message}</div>', unsafe_allow_html=True)


#custom css for button and webview
st.markdown("""
    <style>
    
    .big-font {
    font-size:80px !important;
}

     .message-container {
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        align-self: flex-end;

    }
    .other-message {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        align-self:flex-start;

    }

    .notification-message {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        align-self:center;

    }

    .sender {
        font-size: 12px;
        color: red;
    }
    .time {
        font-size: 10px;
        color: black;
        text-align: right;

    }

       .chat-message {
        background-color: #DCF8C6;
        color: #000000;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        max-width: 70%;
    }
    .chat-message:nth-child(odd) {
        align-self: flex-start;
        text-align: left;
    }
    .chat-message:nth-child(even) {
        align-self: flex-end;
        text-align: right;
        background-color: #DCF8C6;
        color: #000000;
    }



    </style>
""", unsafe_allow_html=True)

github_link = ""
file_path = 'sample_whatsapp_export.txt'


@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = f"""<img style = "width: 100%" src="data:image/svg+xml;base64,{b64}"/>"""
    st.write(html, unsafe_allow_html=True)

def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """



    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    b64 = base64.b64encode(object_to_download.encode()).decode()
    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:text/plain;base64,{b64}">{button_text}</a><br></br>'

    return dl_link

def display_chat_message(sender, message, sentiment):
    st.markdown(
        f"<div class='chat-message'>Sender: {sender}<br>Message: {message}<br>Sentiment: {sentiment}</div>",
        unsafe_allow_html=True
    )



#Options Menu
with st.sidebar:
    selected = option_menu( 'Chat Analyzer', ["Intro", 'Search','About'],icons=['play-btn','search','info-circle'],menu_icon='intersect', default_index=0)
    lottie = load_lottiefile("lottie_jsons/sidebar.json")
    st_lottie(lottie, key='loc')

if selected == "Intro":
    c1, c2 = st.columns((2, 1))
    c1.title("""Whatsapp Chat Analyser""")
    c1.subheader("""Discover trends, analyse your chat history and judge your friends!""")
    c1.markdown(
        f"Dont worry, we wont peek, we're not about that, in fact, you can check the code in here: [link]({github_link})")

    uploaded_file = c1.file_uploader(label="""Upload your Whatsapp chat, don't worry, we won't peek""",
                                     key="notniq")

    with open(file_path, 'r') as f:
        dl_button = download_button(f.read(), 'sample_file.txt', 'Try it out with my sample file!')
        c1.markdown(dl_button, unsafe_allow_html=True)

    with c2:
        lottie = load_lottiefile("lottie_jsons/chat_icon.json")
        st_lottie(lottie, key='loc2')

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)
        print("YOUR DATA FRAME IS: ", df)
        st.write(df)

        # # Assuming 'date' is a datetime column in your DataFrame
        # start_date = df['date'].min()
        # end_date = df.iloc[-1]['date']
        #
        # # Convert datetime to timestamp for slider
        # start_date_timestamp = int(start_date.timestamp())
        # end_date_timestamp = int(end_date.timestamp())
        #
        #
        # # Convert timestamp back to datetime
        # start_datee = datetime.utcfromtimestamp(start_date_timestamp).strftime('%Y-%m-%d')
        # end_datee = datetime.utcfromtimestamp(end_date_timestamp).strftime('%Y-%m-%d')
        #
        # # Convert to datetime objects for showing them in slider
        # date_object1 = datetime.strptime(start_datee, "%Y-%m-%d")
        # date_object2 = datetime.strptime(end_datee, "%Y-%m-%d")
        #
        #
        # # Create a date slider
        # selected_date_range_timestamp = st.slider(
        #     'Select date',
        #     min_value=date_object1,
        #     value=(date_object1, date_object2),
        #     max_value=date_object2,
        #     format="YYYY-MM-DD",  # Display format
        # )
        #
        # # Display the selected date range
        #
        # selected_start_date = selected_date_range_timestamp[0].strftime("%Y-%m-%d")
        # selected_end_date = selected_date_range_timestamp[1].strftime("%Y-%m-%d")
        # st.write(f'Start Date: {selected_date_range_timestamp[0].strftime("%Y-%m-%d")} & End Date: {selected_date_range_timestamp[1].strftime("%Y-%m-%d")}')
        #
        # # Filter data based on selected date range
        # df = df[(df['date'] >= selected_start_date) & (df['date'] <= selected_end_date)]
        # st.write(df)

        # fetch unique users
        user_list = df['user'].unique().tolist()
        # user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        c3, c4, c5 = st.columns((1, 2, 1))

        with c3:
            selected_user = st.selectbox("Select Participants for analysis", user_list, )
        with c4:
            selected_participants = st.multiselect("Select Participants to view there conversation", user_list,
                                                   key="new2")
        with c5:
            selected_participant_for_displaying_messsage = st.selectbox("select participant for viewing: ",selected_participants)

        placeholder = st.empty()

        if selected_user:
            with placeholder.container():
                # Stats Area
                num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
                st.title("Top Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.header("Total Messages")
                    st.title(num_messages)
                with col2:
                    st.header("Total Words")
                    st.title(words)
                with col3:
                    st.header("Media Shared")
                    st.title(num_media_messages)
                with col4:
                    st.header("Links Shared")
                    st.title(num_links)

                # monthly timeline
                st.title("Monthly Timeline")
                timeline = helper.monthly_timeline(selected_user, df)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

                # daily timeline
                st.title("Daily Timeline")
                daily_timeline = helper.daily_timeline(selected_user, df)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

                # activity map
                st.title('Activity Map')
                col1, col2 = st.columns(2)

                with col1:
                    st.header("Most busy day")
                    busy_day = helper.week_activity_map(selected_user, df)
                    fig, ax = plt.subplots()
                    ax.bar(busy_day.index, busy_day.values, color='purple')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                with col2:
                    st.header("Most busy month")
                    busy_month = helper.month_activity_map(selected_user, df)
                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='orange')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                st.title("Weekly Activity Map")
                user_heatmap = helper.activity_heatmap(selected_user, df)
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)

                # finding the busiest users in the group(Group level)
                if selected_user == 'Overall':
                    st.title('Most Busy Users')
                    x, new_df = helper.most_busy_users(df)
                    fig, ax = plt.subplots()

                    col1, col2 = st.columns(2)

                    with col1:
                        ax.bar(x.index, x.values, color='red')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    with col2:
                        st.dataframe(new_df)

                st.title("Wordcloud")
                df_wc = helper.create_wordcloud(selected_user, df)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                plt.axis("off")
                st.pyplot(fig)

                # most common words
                most_common_df = helper.most_common_words(selected_user, df)

                fig, ax = plt.subplots()

                ax.barh(most_common_df[0], most_common_df[1])

                st.title('Most commmon words')
                st.pyplot(fig)

                # emoji analysis
                emoji_df = helper.emoji_helper(selected_user, df)
                st.title("Emoji Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(emoji_df)
                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                    st.pyplot(fig)

                max_user, max_time = helper.longest_reply_user(df)
                st.write(max_user, " takes the most time to reply wiz ", max_time)
                helper.message_length_analysis(selected_user, df)
                helper.busiest_hours_analysis(df)
                helper.message_count_by_month(selected_user, df)
                helper.top_emojis_used(selected_user, df)
                helper.greeting_farewell_analysis(selected_user, df)


                c_11, c_12 = st.columns((1, 1))
                fig1, most_messages_winner = helper.message_count_aggregated_graph(df)
                c_11.subheader("Who talks the most?")
                c_11.markdown(
                    f"How many messages has each one sent in your convo? apparently **{most_messages_winner}** did")
                c_11.pyplot(fig1)

                ##This functionalities are not working

                # c_11, c_12 = st.columns((1, 1))
                # fig1, most_messages_winner = helper.conversation_starter_graph(df)
                # c_11.subheader("Who's starts the conversations?")
                # c_11.markdown(f"This clearly shows that **{most_messages_winner}** started all the convos")
                # c_11.pyplot(fig1)


                # fig, max_message_count, max_message_count_date = helper.create_messages_per_week_graph(df)
                # st.subheader("When did you talk the most?")
                # st.markdown(
                #     f"This is how many messages each one of you have exchanged per **week** between the dates of **{slider[0].strftime('%m/%y')}** and **{slider[1].strftime('%m/%y')}**, the most messages you guys have exchanged in a week was **{max_message_count}** on **{max_message_count_date.strftime('%d/%m/%y')}**")
                # st.pyplot(fig)

                # fig = helper.create_average_wpm_graph(df)
                # st.subheader("How many words do your messages have?")
                # st.markdown(
                #     f"This basically shows how much effort each person puts in each message, the more words per message, the more it feels like the person is putting in real effort")
                # st.pyplot(fig)
                #


        if selected_participant_for_displaying_messsage:
            placeholder.empty()
            st.header(f"You are viewing as {selected_participant_for_displaying_messsage}")

            # Convert the 'date' column to pandas datetime object
            df['date'] = pd.to_datetime(df['date'])

            # Sort the DataFrame by 'date' and 'user'
            df = df.sort_values(by=['date', 'user'])

            # Iterate through messages and display them in a chat-like format
            for _, group in df[df['user'].isin(selected_participants)].iterrows():
                sender = group['user']
                message = group['message']
                time = group['date']
                sentiment = helper.analyze_sentiment(message)  # assuming you have a helper function

                # Apply custom CSS class based on the sender
                if sender == selected_participant_for_displaying_messsage:
                    st.markdown(
                        f'<div class="message-container"><div class="user-message">{message}</div>'
                        f'<div class="time"  >{time} </div></div>',
                        unsafe_allow_html=True
                    )
                elif sender == "group_notification":
                    st.markdown(
                        f'<div class="message-container">'
                        f'<div class="notification-message">{message}</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="message-container"><div class="sender">{sender}</div>'
                        f'<div class="other-message">{message}</div><div class="time"  style="text-align:left " >{time}</div></div>'
                        ,
                        unsafe_allow_html=True
                    )

            selected_participant_for_sentiments = placeholder.multiselect(f"Show Messages and Sentiments", user_list)

            if selected_participant_for_sentiments:
                st.header(f"Message sentiments of {selected_participant_for_sentiments}")
                filtered_df = df[df['user'].isin(selected_participant_for_sentiments)]

                for _, group in filtered_df.iterrows():
                    sender = group['user']
                    message = group['message']
                    sentiment = helper.analyze_sentiment(message)  # assuming you have a helper function
                    display_chat_message(sender, message, sentiment)



if selected == "About":
    st.title('How does the chat analyser work?')
    c1, c2 = st.columns([1, 1])
    c1.header('Data Analysis')
    c1.markdown("""The core of this app is made up of very simple data analysis, such as the aggregation of the 
            time of day the messages were sent, or by counting the number of messages were sent each month, however, 
            calculations such as reply times and number of messages in a conversation aren't so trivial, in which we'll 
            dive deeper right now!""")
    with c2:
        st.write()
        # st_lottie(self.lottie_data, height=300)

    st.subheader("How query is refined ??")
    st.code("""
    
        def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']
     
    """)

    st.write(" ")
    st.subheader("How to create a LLM using Langchain & OpenAI")
    st.code("""
        
        
#Create an OpenAI llm model to read csv file generated from dataframe
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

#Assign system level template
system_msg_template = SystemMessagePromptTemplate.from_template(template="You are now a question answering  chatbot you would be provided with a dataframe which is having a    whatsapp conversation that has messages, particpantsname , date time of message etc etc columns you have all the rights to access and retrieve information from the dataframe you would be answering questions and not be telling user how to retrieve that information programatically you yourself will retrirve the information and answer the question ")

#Human Query (input)
human_msg_template = HumanMessagePromptTemplate.froms_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
    
    """)

    st.write(" ")

    st.header('Conversations')
    st.markdown("""In order to analyse conversations, we need to first find a way to define conversations based 
                on a sequence of individual messages. To do that, let's make our definition of a conversation:""")
    st.info("""Conversations:        
                The event of the exchange of messages between two people during a certain period of time""")
    st.markdown("""To implement that in our messages, I have implemented a code that, if it detects a significant 
                amount of time between two messages, say around 1 hour, it'll define the end of a conversation and the 
                beginning of a new one! Here's the code for it:""")
    image = open('images/ConvDiagram.svg', 'r').read()
    render_svg(image)
    st.write(" ")

    st.code("""def cluster_into_conversations(
            df : pd.DataFrame, 
            inter_conversation_threshold_time: int = 60
            ):
        threshold_time_mins = np.timedelta64(inter_conversation_threshold_time, 'm')

        # This calculates the time between the current message and the previous one
        conv_delta = df.index.values - np.roll(df.index.values, 1)
        conv_delta[0] = 0

        # This detects where the time between messages is higher than the threshold
        conv_changes = conv_delta > threshold_time_mins
        conv_changes_indices = np.where(conv_changes)[0]
        conv_codes = []

        # This encodes each message with its own conversation code
        last_conv_change = 0
        for i, conv_change in enumerate(conv_changes_indices):
            conv_codes.extend([i]*(conv_change - last_conv_change))
            last_conv_change = conv_change

        # This serves to ensure that the conversation codes 
        # and the number of messages are aligned
        conv_codes = pad_list_to_value(conv_codes, len(df), conv_codes[-1])
        conv_changes = pad_list_to_value(conv_changes, len(df), False)

        return conv_codes, conv_changes
            """)
    st.header('Replies')
    st.markdown("""This is basically the same issue as the one we had in the conversations, we first need to 
            define it:""")
    st.info("""Reply:
            The response of one person to the messages sent by the previous one within a conversation""")
    st.markdown("""This is faily easy to implement, I will say that a reply happens when the subject changes 
            within a conversation, here's the code for it!:""")
    image = open('images/ReplyDiagram.svg', 'r').read()
    render_svg(image)
    st.write("")
    st.code("""def find_replies(df : pd.DataFrame):
        # These are sanity checks in order to see if I made any ordering mistakes
        assert('Conv code' in df.columns)
        assert('Conv change' in df.columns)
        assert('Subject' in df.columns)
        # Ordinal encoders will encode each subject with its own number
        message_senders = OrdinalEncoder().fit_transform(df['Subject'].values.reshape(-1,1))
        # This compares the current subject with the previous subject 
        # In a way that computers can optimize
        sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
        sender_changed[0] = False
        # This checks if the reply isn't within a different conversation
        is_reply = sender_changed & ~df['Conv change']
        return is_reply, sender_changed""")
    st.markdown("""Notice how that implies that if a reply is the start of a new conversation, it's not a reply, 
            it's the start of a new conversation. This helps us segregate the replies to only those that happen within a 
            conversation, say, when you two are really **Talking** to each other, which I think is more indicative of the 
            level of interaction you two are having""")
