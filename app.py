# Replace your generate_mistral_response_stream function with this version:

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
        "model": "mistral-small-latest",  # Updated model name
        "messages": messages,
        "temperature": 0.3,
        "stream": True
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload, stream=True)
        
        # Debug logging
        st.write(f"🔍 Debug - Status Code: {response.status_code}")
        if response.status_code != 200:
            st.error(f"❌ API Error: {response.status_code}")
            st.error(f"Response: {response.text}")
            return None
            
        return response
        
    except Exception as e:
        st.error(f"❌ Request failed: {str(e)}")
        return None
