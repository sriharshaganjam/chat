
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import time

# Page config must be first
st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="wide")

# Initialize embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Mistral API setup
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]

HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

# -------------------- PDF Parsing --------------------
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# -------------------- Text Chunking --------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# -------------------- Embedding & Retrieval --------------------
def embed_chunks(chunks):
    embeddings = embedder.encode(chunks)
    return np.array(embeddings)

def get_top_k_chunks(query, chunks, chunk_embeddings, k=5):
    query_embedding = embedder.encode([query])[0]
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_k_idx = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_idx], similarities[top_k_idx]

# -------------------- Enhanced Detection Functions --------------------
def is_greeting(user_input):
    """Detect if the input is a simple greeting"""
    greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    input_lower = user_input.lower().strip()
    return (any(word in input_lower for word in greeting_words) and 
            len(user_input.split()) <= 3)

def is_confusion_expression(user_input):
    """Detect expressions of confusion or uncertainty"""
    confusion_phrases = [
        "i don't know", "i dont know", "don't know", "dont know", "not sure", "i'm not sure", "im not sure",
        "no idea", "unsure", "i don't understand", "don't understand", "i dont understand", "dont understand",
        "i'm confused", "im confused", "confused", "help me", "explain", "i have no idea",
        "no clue", "i'm clueless", "clueless", "lost", "i'm lost", "im lost", "huh", "what", "sorry", "help me with this"
    ]
    input_lower = user_input.lower().strip()
    return any(phrase in input_lower for phrase in confusion_phrases)

def has_educational_context(user_input, previous_context=None, chunks=None, chunk_embeddings=None, similarity_threshold=0.15):
    """
    Check if the confusion is in context of educational material using semantic similarity.
    
    Args:
        user_input: The user's input expressing confusion
        previous_context: Previous conversation context
        chunks: Text chunks from the PDF
        chunk_embeddings: Embeddings of the chunks
        similarity_threshold: Minimum similarity score to consider as educational context
    
    Returns:
        tuple: (is_educational, max_similarity_score)
    """
    # If there's previous context from the conversation, it's likely educational
    if previous_context:
        return True, 1.0
    
    # If we don't have PDF content loaded, we can't determine context
    if chunks is None or chunk_embeddings is None:
        return False, 0.0
    
    # Use semantic similarity to check if the confusion relates to course content
    try:
        # Get embedding for the user input
        input_embedding = embedder.encode([user_input])[0]
        
        # Calculate similarities with all chunks
        similarities = np.dot(chunk_embeddings, input_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(input_embedding)
        )
        
        max_similarity = np.max(similarities)
        
        # If the confusion has reasonable similarity to course content, it's educational
        is_educational = max_similarity >= similarity_threshold
        
        return is_educational, max_similarity
        
    except Exception as e:
        # If there's any error in similarity calculation, default to non-educational
        return False, 0.0

def get_confusion_context_chunks(user_input, chunks, chunk_embeddings, k=3):
    """
    Get the most relevant chunks for confusion context, specifically for supportive responses.
    This helps provide targeted help for the confused student.
    """
    if chunks is None or chunk_embeddings is None:
        return [], []
    
    try:
        input_embedding = embedder.encode([user_input])[0]
        similarities = np.dot(chunk_embeddings, input_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(input_embedding)
        )
        top_k_idx = similarities.argsort()[-k:][::-1]
        return [chunks[i] for i in top_k_idx], similarities[top_k_idx]
    except:
        return [], []

# -------------------- Mistral Streaming Response --------------------
def generate_mistral_response_stream(context, user_question, chat_history, response_type="normal"):
    # Build conversation history for context
    if response_type == "confusion_with_context":
        system_content = f"""You are a supportive AI tutor helping a student who has expressed confusion or uncertainty. The student needs encouragement and gentle guidance.

CRITICAL RULES:
1. The student has indicated they don't know something or are confused - this is NORMAL and part of learning
2. Provide hints, clues, and encouragement to guide them toward understanding
3. Break down complex topics into smaller, manageable pieces
4. Use the course content to provide specific guidance and examples
5. Be patient, supportive, and encouraging
6. Ask guiding questions to help them think through the problem
7. Never make them feel bad for not knowing - confusion is part of learning!

Course Content (most relevant to their confusion):
{context}

The student is expressing uncertainty about something related to the course material - help them learn step by step."""
    else:
        system_content = f"""You are a strict AI tutor that ONLY uses the provided course content to help students. 

CRITICAL RULES:
1. You can ONLY discuss topics that are directly mentioned in the provided course content
2. If a question is outside the scope of the uploaded content, respond warmly but redirect: "I'd love to help you with that! However, I can only assist with topics that are covered in your uploaded course material."
3. Use the provided course content to guide students with hints and questions - do NOT give direct answers
4. Be encouraging and supportive, but stay strictly within the bounds of the uploaded material
5. Never hallucinate or make up information not present in the course content

Course Content:
{context}"""
    
    messages = [{"role": "system", "content": system_content}]
    
    # Add chat history
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current question
    messages.append({"role": "user", "content": user_question})

    payload = {
        "model": "mistral-medium",
        "messages": messages,
        "temperature": 0.3,
        "stream": True
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload, stream=True)
        if response.status_code == 200:
            return response
        else:
            return None
    except Exception as e:
        return None

# -------------------- Chat History Management --------------------
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "last_topic_context" not in st.session_state:
        st.session_state.last_topic_context = None

def add_to_chat_history(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def display_chat_history():
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

def get_conversation_context():
    """Get recent conversation context to understand if confusion is in educational context"""
    if len(st.session_state.chat_history) >= 2:
        # Look at the last few exchanges
        recent_messages = st.session_state.chat_history[-4:]  # Last 4 messages
        recent_text = " ".join([msg["content"] for msg in recent_messages])
        return recent_text
    return None

# -------------------- Main Streamlit UI --------------------
st.title("🎓 AI Tutor - Learn with Guidance")

# Initialize session state
initialize_session_state()

# Sidebar for PDF upload
with st.sidebar:
    st.header("📚 Course Material")
    uploaded_file = st.file_uploader("Upload your course PDF", type=["pdf"])
    
    # Fixed similarity threshold
    similarity_threshold = 0.15
    
    if uploaded_file:
        if st.session_state.chunks is None:
            with st.spinner("Processing your PDF..."):
                full_text = extract_text_from_pdf(uploaded_file)
                st.session_state.chunks = chunk_text(full_text)
                st.session_state.embeddings = embed_chunks(st.session_state.chunks)
            st.success("✅ PDF processed successfully!")
        else:
            st.success("✅ PDF ready!")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_topic_context = None
        st.rerun()

# Main chat interface
if uploaded_file and st.session_state.chunks is not None:
    st.markdown("### 💬 Chat with your AI Tutor")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your course material...")
    
    if user_input:
        # Add user message to chat history
        add_to_chat_history("user", user_input)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Check different types of input
        is_greeting_input = is_greeting(user_input)
        is_confusion_input = is_confusion_expression(user_input)
        conversation_context = get_conversation_context()
        
        # Use semantic similarity to determine educational context
        has_edu_context, edu_similarity_score = has_educational_context(
            user_input, 
            conversation_context, 
            st.session_state.chunks, 
            st.session_state.embeddings,
            similarity_threshold
        )
        
        # Get general content relevance for regular questions
        top_chunks, similarities = get_top_k_chunks(
            user_input, 
            st.session_state.chunks, 
            st.session_state.embeddings
        )
        max_similarity = max(similarities) if len(similarities) > 0 else 0
        
        # Decision logic with improved semantic understanding
        if is_greeting_input:
            response_text = "Hello! I'm excited to help you explore and understand your course material. I'm here to guide you through the content with questions and hints to help you learn effectively. What topic from your uploaded material would you like to dive into?"
            with st.chat_message("assistant"):
                st.write(response_text)
            add_to_chat_history("assistant", response_text)
            
        elif is_confusion_input and has_edu_context:
            # Student is confused about something related to course content - provide targeted guidance
            confusion_chunks, confusion_similarities = get_confusion_context_chunks(
                user_input, 
                st.session_state.chunks, 
                st.session_state.embeddings
            )
            
            context = "\n\n".join(confusion_chunks) if confusion_chunks else "\n\n".join(top_chunks)
            st.session_state.last_topic_context = context
            
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                response_stream = generate_mistral_response_stream(
                    context, 
                    user_input, 
                    st.session_state.chat_history[:-1],
                    response_type="confusion_with_context"
                )
                
                if response_stream and response_stream.status_code == 200:
                    for line in response_stream.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                line = line[6:]
                                if line.strip() == '[DONE]':
                                    break
                                try:
                                    json_data = json.loads(line)
                                    if 'choices' in json_data and len(json_data['choices']) > 0:
                                        delta = json_data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            full_response += delta['content']
                                            message_placeholder.write(full_response + "▌")
                                            time.sleep(0.01)
                                except json.JSONDecodeError:
                                    continue
                    message_placeholder.write(full_response)
                else:
                    full_response = "That's okay! Let me help you understand this step by step. Don't worry - asking questions and expressing confusion is a natural part of learning. Let me guide you through the relevant concepts from your course material."
                    message_placeholder.write(full_response)
                
                add_to_chat_history("assistant", full_response)
                
        elif max_similarity >= 0.1:  # Question relates to course content
            context = "\n\n".join(top_chunks)
            st.session_state.last_topic_context = context
            
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                response_stream = generate_mistral_response_stream(
                    context, 
                    user_input, 
                    st.session_state.chat_history[:-1]
                )
                
                if response_stream and response_stream.status_code == 200:
                    for line in response_stream.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                line = line[6:]
                                if line.strip() == '[DONE]':
                                    break
                                try:
                                    json_data = json.loads(line)
                                    if 'choices' in json_data and len(json_data['choices']) > 0:
                                        delta = json_data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            full_response += delta['content']
                                            message_placeholder.write(full_response + "▌")
                                            time.sleep(0.01)
                                except json.JSONDecodeError:
                                    continue
                    message_placeholder.write(full_response)
                else:
                    full_response = "I apologize, but I'm having trouble processing your request right now. Please try again."
                    message_placeholder.write(full_response)
                
                add_to_chat_history("assistant", full_response)
                
        else:
            # Question is outside course scope or confusion is not educational
            if is_confusion_input:
                response_text = f"I understand you're feeling uncertain, but this seems to be outside the scope of your uploaded course material. I can only help with topics covered in your PDF. Feel free to ask me about any concepts, theories, or topics from your course material!"
            else:
                response_text = "I'd love to help you with that! However, I can only assist with topics that are covered in your uploaded course material. This question seems to be outside the scope of what we've covered together. What would you like to explore from your course content?"
            
            with st.chat_message("assistant"):
                st.write(response_text)
            add_to_chat_history("assistant", response_text)

else:
    st.info("👆 Please upload a course PDF in the sidebar to begin learning!")
    st.markdown("""
    ### How to use this ✨AI Tutor:
    
    1. **Upload your course material** (PDF) using the sidebar
    2. **Ask questions** about the content in the chat
    3. **Express confusion freely** - the tutor uses AI to understand if your confusion relates to the course material
    4. **Get personalized guidance** - the tutor provides targeted help based on the most relevant parts of your material
    
    
    **Note:** Your chat history is session-based and will be cleared when you close the browser.
    """)
