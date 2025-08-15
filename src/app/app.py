import logging
import os
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
w = WorkspaceClient()

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

def get_user_info():
    headers = st.context.headers
    user_access_token = headers.get("X-Forwarded-Access-Token")
    user_name=headers.get("X-Forwarded-Preferred-Username")
    user_display_name = ""
    if user_access_token:
        # Initialize WorkspaceClient with the user's token
        w = WorkspaceClient(token=user_access_token, auth_type="pat")
        # Get current user information
        current_user = w.current_user.me()
        # Display user information
        user_display_name = current_user.display_name
        

    return dict(
        user_name=user_name,
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
        user_access_token = headers.get("X-Forwarded-Access-Token"),
        user_display_name = user_display_name if user_display_name != "" else user_name
    )

user_info = get_user_info()

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

c1, c2 = st.columns([1,1])
with c1:
    st.markdown("### :material/coronavirus: CoviVader ")
with c2:
    st.write(f"Hello, {user_info['user_display_name']}")

st.markdown("#### May the Facts Be With You")
st.write(f"I can help you with questions related to Studies on COVID-19 Clinical Trials.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How many trials were conducted in New York?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [ChatMessage(role=ChatMessageRole.USER, content=prompt)]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Query the Databricks serving endpoint
        try:
            response = w.serving_endpoints.query(
                name=os.getenv("SERVING_ENDPOINT"),
                messages=messages,
                max_tokens=400,
            )
            assistant_response = response.choices[0].message.content

            st.markdown(assistant_response)
        except Exception as e:
            st.error(f"Error querying model: {e}")
            assistant_response="An error occured"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
