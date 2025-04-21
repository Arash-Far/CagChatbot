import json
import os
import sqlite3
import uuid
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, Literal

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader

# LangGraph imports
from langgraph.graph import START, MessagesState, StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver



# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup and connection handling
DB_PATH = "database.sqlite"
CHECKPOINT_DB_PATH = "checkpoint.sqlite"  # Separate database for LangGraph checkpointer

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_checkpoint_connection():
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    return conn

# Initialize database tables if they don't exist
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create conversations table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')
    
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

# Load PDF documents
def load_products():
    """Load products from the database and format them for the knowledge base"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all products
        cursor.execute("""
            SELECT id, name, description, price, created_at
            FROM products
            ORDER BY name
        """)
        
        products = cursor.fetchall()
        conn.close()
        
        if not products:
            print("No products found in database")
            return ""
        
        # Format products into readable text
        products_text = "Available Products:\n\n"
        for product in products:
            products_text += f"- {product['name']}\n"
            products_text += f"  Description: {product['description']}\n"
            products_text += f"  Price: ${product['price']:.2f}\n\n"
        
        return products_text
        
    except Exception as e:
        print(f"Error loading products: {str(e)}")
        return ""

def load_knowledge_files():
    """Load content from knowledge text files and products"""
    try:
        with open("knowledge/FAQs.txt", "r") as faqs_file:
            faqs_content = faqs_file.read()
        
        with open("knowledge/Referral_Rules.txt", "r") as rules_file:
            rules_content = rules_file.read()
        
        products_content = load_products()
        
        return f"FAQs:\n{faqs_content}\n\nReferral Rules:\n{rules_content}\n\nProduct Catalog:\n{products_content}"
    except Exception as e:
        print(f"Error loading knowledge files: {str(e)}")
        return ""

knowledge_content = load_knowledge_files()

# Initialize OpenAI model
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

model = ChatOpenAI(
    model="gpt-4.1-mini",  # Using GPT-4.1 mini as specified in instructions
    api_key=api_key,
    streaming=True
)

# LangGraph State class
class State(MessagesState):
    user_id: int
    conversation_id: Optional[str] = None
    knowledge_context: str = ""

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Capper, a friendly, concise AI assistant designed to help users quickly achieve their goals within the Carton Caps app. Your primary functions include recommending products and assisting users with the referral program, both geared towards benefiting schools through user engagement.

Tone & Clarity:
Maintain a friendly, approachable, and concise tone. Your responses must be clear, easy to understand, and build user trust. Emphasize how user actions positively impact schools.

Goal-Oriented Flow:
Your aim is to help users complete their objectives—such as getting personalized product recommendations or understanding the referral process—in no more than 2-3 conversational exchanges. Always gently guide conversations towards resolution.

Content Filtering:
Refuse politely and redirect users when receiving off-topic, inappropriate, or abusive questions. Clearly communicate when a request is beyond your capabilities, recommending human support when needed.

Context Awareness:
Always leverage the provided conversation history (past user messages and your responses) to maintain context and ensure a coherent and relevant dialogue.

Fallback Behavior:
If uncertain about the user's intent, politely ask for clarification or rephrasing. If the uncertainty persists or you detect frustration, seamlessly offer to escalate to human support.

Specific Knowledge Context:
1. Product Recommendations:
- Recommend products tailored to user preferences or dietary restrictions (e.g., nut-free, dairy-free).
- Clearly highlight current bonus products offering extra "Carton Caps" points.
- Suggest simple, appealing products suitable for picky eaters, verifying dietary restrictions and specific dislikes.

2. Referral Program:
- Clearly explain how users can refer friends using unique referral codes or links.
- Walk users through the referral process step-by-step, including downloading the app, creating an account, and ensuring referral codes are used correctly.
- Clarify referral bonuses, qualifying conditions, eligibility restrictions, and potential troubleshooting issues such as delays in receiving bonuses or missed referral links.
- Firmly refuse activities violating referral program policies (self-referrals, fraudulent practices).

Privacy and Security Guardrails:
- Never disclose sensitive personal data or information belonging to other users. Politely decline requests for such information.
- Stick strictly to Carton Caps-related topics, clearly redirecting or declining irrelevant discussions.

Context: {context}
"""
    ),
    MessagesPlaceholder(variable_name="messages"),
])



# LangGraph nodes
def retrieve_context(state: State):
    """Node to retrieve user context and knowledge"""
    # Add PDF content as context
    return {"knowledge_context": knowledge_content}

def generate_response(state: State) -> State:
    messages = state["messages"]
    # Use the global prompt variable
    prompt_messages = prompt.invoke({"messages": messages, "context": state["knowledge_context"]})
    response = model.invoke(prompt_messages)
    return {"messages": [response]}

# Build the graph
def build_graph():
    # Create a new connection for the checkpointer
    checkpoint_conn = get_checkpoint_connection()
    
    try:
        print("Creating SQLite checkpointer...")
        # Create the SQLite checkpointer with the connection
        checkpointer = SqliteSaver(checkpoint_conn)
        
        print("Building StateGraph...")
        # Build the graph
        graph = StateGraph(State)
        
        print("Adding nodes...")
        # Add nodes
        graph.add_node("retrieve_context", retrieve_context)
        graph.add_node("generate_response", generate_response)
        
        print("Adding edges...")
        # Add edges
        graph.add_edge(START, "retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")
        graph.add_edge("generate_response", END)
        
        print("Compiling graph...")
        return graph.compile(checkpointer=checkpointer)
    except Exception as e:
        print(f"Error building graph: {str(e)}")
        # Close the connection if there's an error
        checkpoint_conn.close()
        raise e


def create_user(name, email):
    """Create a new user with the given name and email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Generate a unique ID for the user
    import hashlib
    hash_object = hashlib.md5(email.encode())
    user_id = int(hash_object.hexdigest()[:8], 16)
    
    # Get current timestamp for created_at
    from datetime import datetime
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Insert the user into the database
    cursor.execute("INSERT INTO users (id, name, email, created_at) VALUES (?, ?, ?, ?)", 
                  (user_id, name, email, created_at))
    conn.commit()
    conn.close()
    
    return user_id

def get_or_create_user(name, email):
    """Get an existing user by email or create a new one if not found."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user exists with the given email
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    
    if result:
        # User exists, return the ID
        user_id = result[0]
        conn.close()
        return user_id
    else:
        # User doesn't exist, create a new one
        conn.close()
        return create_user(name, email)

def create_conversation(user_id: int, conversation_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO conversations (id, user_id) VALUES (?, ?)",
        (conversation_id, user_id)
    )
    
    conn.commit()
    conn.close()
    
    return conversation_id



def get_conversation_messages(conversation_id: str):
    """Get conversation messages in LangChain format using LangGraph's state history"""
    try:
        # Create a new connection for this call
        conn = get_checkpoint_connection()
        
        # Build a graph to access its state history
        graph = build_graph()
        
        # Configuration with the conversation ID as thread_id
        config = {"configurable": {"thread_id": conversation_id}}
        
        # Retrieve the state history
        history = list(graph.get_state_history(config))
        
        # Extract messages from the history
        messages = []
        for snapshot in history:
            if "messages" in snapshot.values:
                for msg in snapshot.values["messages"]:
                    if msg not in messages:  # Avoid duplicates
                        messages.append(msg)
        
        # Close the connection
        conn.close()
        
        return messages
    except Exception as e:
        print(f"Error retrieving conversation history: {str(e)}")
        # Return an empty list if there's an error
        return []

def verify_conversation_owner(conversation_id: str, user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT user_id FROM conversations WHERE id = ?",
        (conversation_id,)
    )
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return False
    
    return result["user_id"] == user_id


# API endpoints

@app.post("/get_or_create_user")
async def get_or_create_user_endpoint(request: Request):
    try:
        # Parse the request body
        body = await request.json()
        name = body.get("name")
        email = body.get("email")
        
        if not name or not email:
            raise HTTPException(status_code=400, detail="Name and email are required")
        
        # Get or create the user
        user_id = get_or_create_user(name, email)
        
        # Return the user ID
        return {"userId": user_id}
    except Exception as e:
        print(f"Error getting or creating user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat")
async def chat_endpoint(request: Request, stream: bool = False):
    # Parse the request

    try:
        if request.query_params.get("query"):
            # Handle client.py requests (where query is passed as query param)
            query = request.query_params.get("query")
            # For simplicity in POC, use a default user ID
            user_id = 1
            conversation_id = None
            message = query
        else:
            # Handle regular API requests as defined in the API contract
            body = await request.json()
            user_id = body.get("userId")
            conversation_id = body.get("conversationId")
            message = body.get("message")
            
            if not user_id or not message:
                raise HTTPException(status_code=400, detail="Missing required fields")
    except Exception as e:
        print(f"Error parsing request: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
        
   
    
    # Create new conversation if needed
    if not conversation_id:
        conversation_id = f"c_{uuid.uuid4().hex[:12]}"
        create_conversation(user_id, conversation_id)
    else:
        # Verify conversation ownership
        if not verify_conversation_owner(conversation_id, user_id):
            raise HTTPException(status_code=403, detail="Forbidden")
    
    
    # Get existing messages or initialize with new message
    messages = get_conversation_messages(conversation_id)
    if not messages or messages[-1].type != "human":
        messages = [HumanMessage(content=message)]
    
    # Create a new connection for the checkpointer
    checkpoint_conn = get_checkpoint_connection()
    
    try:
        # Build the graph
        graph = build_graph()
        
        # Initial state
        initial_state = {
            "messages": messages,
            "user_id": user_id,
            "conversation_id": conversation_id
        }
        print(f"Initial state: {initial_state}")
        
        # Configuration
        config = {"configurable": {"thread_id": conversation_id}}
        print(f"Config: {config}")
        
        
        if stream:
            # Streaming response
            async def generate_stream():
                try:
                    full_response = ""
                    for chunk in graph.stream(initial_state, config, stream_mode="messages"):
                        if isinstance(chunk, tuple):
                            message_chunk = chunk[0]
                            if message_chunk.content:
                                yield f'data: {json.dumps({"delta": message_chunk.content})}\n\n'
                                full_response += message_chunk.content
                    
                    # After completion, return full response with metadata
                    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                
                    yield f'data: {json.dumps({"done": True, "botResponse": full_response, "conversationId": conversation_id, "timestamp": timestamp})}\n\n'
                    
                except Exception as e:
                    print(f"Error in stream generation: {str(e)}")
                    yield f'data: {json.dumps({"error": str(e)})}\n\n'
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Blocking response
            try:
                print("Invoking graph...")
                result = graph.invoke(initial_state, config)
                print(f"conversationId: {conversation_id}")
                # Extract the AI message content from the result
                bot_response = result["messages"][-1].content if result["messages"] else ""

                print(f"bot_response: {bot_response}")
                timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                
                return {
                    "status": "success",
                    "botResponse": bot_response,
                    "conversationId": conversation_id,
                    "timestamp": timestamp
                }
            except Exception as e:
                print(f"Error invoking graph: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always close the checkpoint connection
        checkpoint_conn.close()

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request = None):
    # Extract user_id from JWT (in a real system)
    # In this POC, we'll assume it's passed as a header
    user_id = request.headers.get("user-id")
    print("user_id: ", user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        user_id = int(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    # Verify conversation ownership
    if not verify_conversation_owner(conversation_id, user_id):
        raise HTTPException(status_code=404, detail="Not found")
    
    # Get conversation history
    history = get_conversation_messages(conversation_id)
    
    # Format the messages for client consumption
    formatted_messages = []
    for msg in history:
        # Extract message type and content
        msg_type = "user" if msg.type == "human" else "assistant"
        msg_content = msg.content
        
        # Create a formatted message dictionary
        formatted_msg = {
            "role": msg_type,
            "content": msg_content
        }
        
        formatted_messages.append(formatted_msg)

    print("formatted_messages: ", formatted_messages)
    return formatted_messages

@app.get("/user/{user_id}/conversations")
async def get_user_conversations(user_id: int):
    """Get all conversation IDs for a specific user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all conversations for the user
        cursor.execute(
            "SELECT id, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        
        conversations = cursor.fetchall()
        conn.close()
        
        # Format the response
        result = []
        for conv in conversations:
            result.append({
                "conversationId": conv["id"],
                "createdAt": conv["created_at"]
            })
        
        return result
    except Exception as e:
        print(f"Error getting user conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Carton Caps AI Assistant API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)