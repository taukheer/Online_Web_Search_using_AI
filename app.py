import streamlit as st
from duckduckgo_search import DDGS
from swarm import Swarm, Agent
from datetime import datetime

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Initialize Swarm client
client = Swarm()

# 1. Create Internet Search Tool
def get_news_articles(topic):
    # DuckDuckGo search
    ddg_api = DDGS()
    results = ddg_api.text(f"{topic}", max_results=10)
    if results:
        news_results = "\n\n".join([f"Title: {result['title']}\nURL: {result['href']}\nDescription: {result['body']}" for result in results])
        return news_results
    else:
        return f"Could not find news results for {topic}."

# 2. Create AI Agents

# News Agent to fetch news
news_agent = Agent(
    name="News Assistant",
    instructions="You provide the latest news articles for a given topic using DuckDuckGo search. Search for the exact keywords and return results matching the search phrase. Do not hallucinate and do not provide your own opinion.",
    functions=[get_news_articles],
    model="llama3.2"
)

# Editor Agent to summarize the news
editor_agent = Agent(
    name="Editor Assistant",
    instructions="Based on the findings, please provide complete details about the topic based strictly on the details provided. Do not make up content. I am using these findings for a sales pitch, so accuracy is important.",
    model="llama3.2"
)

# 3. Create workflow function
def run_news_workflow(topic, conversation_history):
    # Step 1: Fetch news
    news_response = client.run(
        agent=news_agent,
        messages=conversation_history + [{"role": "user", "content": f"{topic}"}],
    )
    raw_news = news_response.messages[-1]["content"]

    # Step 2:
        # Step 2: Pass news to editor for summarization
    summarized_news_response = client.run(
        agent=editor_agent,
        messages=conversation_history + [{"role": "user", "content": raw_news}],
    )

    # Return the final summarized news
    return summarized_news_response.messages[-1]["content"]

# Streamlit app setup
st.title("Conversational Summary App")
st.write("Ask a question about any topic and get a summarized report. You can also ask follow-up questions.")

# Initialize session state for conversation history if not already done
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Input for user topic
user_input = st.text_input("Enter a topic or question:")

# Button to run the workflow
if st.button("Get Summary"):
    if user_input:
        # Update conversation history immediately after user input
        st.session_state.conversation_history.append({"role": "user", "content": user_input})

        with st.spinner("Fetching and summarizing data..."):
            # Run the news workflow, maintaining the conversation history
            summary = run_news_workflow(user_input, st.session_state.conversation_history)

            # Update conversation history with the assistant's response
            st.session_state.conversation_history.append({"role": "assistant", "content": summary})

            # Display the result
            st.write("### Summary:")
            st.write(summary)
    else:
        st.warning("Please enter a topic or question.")

# Option to clear conversation history
if st.button("Clear History"):
    st.session_state.conversation_history = []
    st.success("Conversation history cleared.")

# Display conversation history
st.write("## Conversation History")
for idx, msg in enumerate(st.session_state.conversation_history):
    role = "User" if msg['role'] == 'user' else "Assistant"
    st.write(f"**{role}:** {msg['content']}")

