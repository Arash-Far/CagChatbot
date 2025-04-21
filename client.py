import streamlit as st
import requests
import json
import sseclient
import time

# Define the API URL
API_URL = "http://localhost:8000"  # FastAPI server address

def stream_response(response):
    """Handle streaming response from the server"""
    message_placeholder = st.empty()
    full_response = ""
    
    # Check if this is a streaming response or a regular JSON response
    content_type = response.headers.get('Content-Type', '')
    
    if 'text/event-stream' in content_type:
        # Handle streaming response
        for line in response.iter_lines():
            if line:
                # Decode the line and remove "data: " prefix
                try:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove "data: " prefix
                        
                        # Parse the JSON data
                        try:
                            json_data = json.loads(data)
                            
                            # Handle different response formats
                            if "delta" in json_data:
                                # Streaming delta update
                                delta = json_data["delta"]
                                full_response += delta
                                message_placeholder.markdown(full_response + "▌")
                            elif "done" in json_data and json_data["done"]:
                                # Final response with metadata
                                full_response = json_data["botResponse"]
                                # Store conversation ID if provided
                                if "conversationId" in json_data:
                                    st.session_state["conversation_id"] = json_data["conversationId"]
                                message_placeholder.markdown(full_response)
                        except json.JSONDecodeError:
                            # Handle plain text responses
                            full_response += data
                            message_placeholder.markdown(full_response + "▌")
                except Exception as e:
                    st.error(f"Error processing stream: {str(e)}")
                    break
    else:
        # Handle regular JSON response
        try:
            json_data = response.json()
            if "botResponse" in json_data:
                full_response = json_data["botResponse"]
                # Store conversation ID if provided
                if "conversationId" in json_data:
                    st.session_state["conversation_id"] = json_data["conversationId"]
                message_placeholder.markdown(full_response)
            else:
                full_response = str(json_data)
                message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error processing response: {str(e)}")
            full_response = "Error processing response"
    
    message_placeholder.markdown(full_response)
    return full_response

def handle_registration(email):
    """Handle user registration process"""
    if not email:
        st.warning("Please enter your email.")
        return False
        
    try:
        response = requests.post(
            f"{API_URL}/get_or_create_user",
            json={"name": email, "email": email}
        )
        
        if response.status_code != 200:
            st.error(f"Registration failed: {response.text}")
            return False
            
        user_data = response.json()
        st.session_state["user_id"] = user_data["userId"]
        st.session_state["user_name"] = email
        st.session_state["user_email"] = email
        
        # Fetch initial conversations
        fetch_user_conversations(user_data["userId"])
        st.success(f"Welcome! You're now registered.")
        st.rerun()
        return True
        
    except Exception as e:
        st.error(f"Error during registration: {str(e)}")
        return False

def fetch_user_conversations(user_id):
    """Fetch conversations for a user"""
    try:
        conv_response = requests.get(f"{API_URL}/user/{user_id}/conversations")
        if conv_response.status_code == 200:
            st.session_state["conversations"] = conv_response.json()
    except Exception as e:
        print(f"Error fetching conversations: {str(e)}")

def load_conversation_history(conv_id):
    """Load history for a specific conversation"""
    try:
        conv_response = requests.get(
            f"{API_URL}/conversation/{conv_id}",
            headers={"user-id": str(st.session_state["user_id"])}
        )
        if conv_response.status_code != 200:
            st.error(f"Failed to load conversation: {conv_response.text}")
            return False
            
        conv_data = conv_response.json()
        st.session_state.messages = []
        for msg in conv_data:
            st.session_state.messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return True
        
    except Exception as e:
        st.error(f"Error loading conversation: {str(e)}")
        return False

def handle_conversation_selection(selected_conv):
    """Handle selection of a conversation from dropdown"""
    if selected_conv == "New Conversation":
        if "conversation_id" in st.session_state:
            del st.session_state["conversation_id"]
        st.session_state.messages = []
        return
        
    conv_id = selected_conv.split(" ")[1]
    if conv_id == st.session_state.get("conversation_id"):
        return
        
    st.session_state["conversation_id"] = conv_id
    if load_conversation_history(conv_id):
        st.rerun()

def handle_logout():
    """Handle user logout"""
    for key in ["user_id", "user_name", "user_email", "conversations", "conversation_id"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def handle_chat_message(prompt, streaming_enabled):
    """Handle sending and receiving chat messages"""
    had_conversation_id = "conversation_id" in st.session_state
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/chat?stream={streaming_enabled}",
                json={
                    "userId": st.session_state["user_id"],
                    "message": prompt,
                    "conversationId": st.session_state.get("conversation_id", None)
                },
                stream=streaming_enabled
            )
            response.encoding = 'utf-8'
            
            full_response = stream_response(response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Refresh conversations list if this was a new conversation
            if not had_conversation_id and "conversation_id" in st.session_state:
                fetch_user_conversations(st.session_state["user_id"])
                st.rerun()
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def render_sidebar():
    """Render the sidebar content"""
    with st.sidebar:
        if "user_id" not in st.session_state:
            with st.form("user_registration"):
                email = st.text_input("Email")
                if st.form_submit_button("Register"):
                    handle_registration(email)
            return
            
        st.write(f"Logged in as: {st.session_state.get('user_name', '')}")
        st.subheader("Your Conversations")
        
        if not st.session_state.get("conversations"):
            st.info("No conversations found. Start a new chat to create one.")
            
        else:
            conv_options = ["New Conversation"] + [
                f"Conversation {conv['conversationId']}" 
                for conv in st.session_state["conversations"]
            ]
            
            current_conv_id = st.session_state.get("conversation_id")
            selected_index = next(
                (i for i, opt in enumerate(conv_options) 
                if current_conv_id and current_conv_id in opt),
                0
            )
            
            selected_conv = st.selectbox(
                "Select a conversation",
                conv_options,
                index=selected_index
            )
            handle_conversation_selection(selected_conv)
        
        if st.button("Logout"):
            handle_logout()
        
        streaming_enabled = st.toggle("Enable Streaming", value=True)
        st.write(f"Streaming mode: {'Enabled' if streaming_enabled else 'Disabled'}")
        return streaming_enabled

def main():
    st.title("Carton Caps Virtual Assistant")
    st.write("Chat with our AI assistant to get product recommendations and learn about our referral program and FAQs!")
    
    streaming_enabled = render_sidebar()
    
    if "user_id" not in st.session_state:
        return
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you today?"):
        handle_chat_message(prompt, streaming_enabled)

if __name__ == "__main__":
    main() 