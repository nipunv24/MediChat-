# backend code
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import io
import os
import time
from PIL import Image
import easyocr
from langgraph.graph import StateGraph, END, START
import torch
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import re # Import re for regex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load environment variables and configure GenAI
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found in environment variables!")
    # In a production app, you might want to stop or raise an exception here

# FastAPI app setup
app = FastAPI(title="MediChat API", description="API for medicine identification and health Q&A chatbot")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Be more restrictive in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OCR Reader Initialization
try:
    ocr_reader = easyocr.Reader(['en'])
    logger.info("EasyOCR reader initialized.")
except Exception as e:
     logger.error(f"Failed to initialize EasyOCR: {e}")
     ocr_reader = None # Handle initialization failure

# Gemini Model Initialization - Using the model name provided by the user
gemini_model = None
if gemini_api_key:
    try:
        # User provided model name "gemini-2.0-flash"
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        logger.info("Gemini model initialized with 'gemini-2.0-flash'.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model 'gemini-2.0-flash': {e}")
        # If the specific model fails, try a common alias as a fallback (optional)
        try:
            logger.info("Trying fallback model 'gemini-1.5-flash-latest'.")
            gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            logger.info("Gemini model initialized with fallback 'gemini-1.5-flash-latest'.")
        except Exception as fallback_e:
             logger.error(f"Failed to initialize fallback model 'gemini-1.5-flash-latest': {fallback_e}")
             gemini_model = None # Both models failed

# --- Pydantic Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    image: Optional[str] = None # Base64 encoded image string

class ChatResponse(BaseModel):
    response: str
    detected_medicines: List[str] = []

# --- LangGraph State and Nodes ---

# Define the state for the LangGraph
class GraphState(dict):
    current_message: str
    history: List[ChatMessage]
    image: Optional[str] = None
    assistant_response: Optional[str] = None
    detected_medicines: List[str] = [] # Keep for frontend highlighting
    prescription_details: List[str] = [] # Keep relevant lines from OCR
    extracted_text: Optional[str] = None # Full text extracted from image


# Define the routing function based on image presence
def route_step(state: Dict[str, Any]) -> str:
    """Routes the flow based on whether an image is present in the state."""
    if state.get("image"):
        logger.info("Routing to medicine_extraction (image detected).")
        return "medicine_extraction"
    else:
        logger.info("Routing to health_expert (no image detected).")
        return "health_expert"


def extract_medicines(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Executing medicine_extraction node.")
    if not ocr_reader:
        logger.error("OCR reader not initialized.")
        state["assistant_response"] = "OCR system is not available. Cannot process image."
        state["detected_medicines"] = []
        state["prescription_details"] = []
        # Return early if OCR is not available
        return state # <--- Indentation level 1 (matching 'def')

    if not gemini_model:
        logger.error("Gemini model not initialized.")
        state["assistant_response"] = "AI model not available for medicine identification."
        state["detected_medicines"] = []
        state["prescription_details"] = []
        # Return early if LLM is not available
        return state # <--- Indentation level 1

    # This node only runs if an image is present due to routing
    image_data = state["image"]
    try: # <--- Outer try block, Indentation level 2
        # --- OCR Processing ---
        # Decode image
        image_data_clean = image_data.split(",")[1] if "," in image_data else image_data
        image_bytes = base64.b64decode(image_data_clean)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Perform OCR
        results = ocr_reader.readtext(image_np)
        extracted_text = " ".join([result[1] for result in results if result[1]]).strip() # Join non-empty, trim whitespace
        state["extracted_text"] = extracted_text
        logger.info(f"OCR Extracted Text: '{extracted_text[:200]}{'...' if len(extracted_text) > 200 else ''}'") # Log first 200 chars

        # Extract potential relevant lines (basic heuristic)
        prescription_details = [res[1].strip() for res in results if res[1].strip() and (res[1].strip()[0].isupper() or len(res[1].split()) > 1)] # Changed split() condition
        state["prescription_details"] = prescription_details[:15] # Increased limit slightly
        logger.info(f"Potential Prescription Lines: {state['prescription_details']}")

        if not extracted_text:
            state["assistant_response"] = "I could not extract any readable text from the image. Please try again with a clearer image."
            state["detected_medicines"] = []
            state["prescription_details"] = []
            # Return early if no text was extracted
            return state # <--- Indentation level 3 (inside outer try, inside if)

        # --- Use LLM for Identification ---
        # Prompt the LLM to identify medications and format the response

        llm_prompt = f"""You are an AI assistant specialized in reading medical prescriptions and documents.
Analyze the following extracted text from an image.
Identify potential medication names (brand or generic), dosages, or key instructions.
Provide a clear summary of what you found based *only* on the text provided. Do not include information not in the text.
If you find nothing resembling a medication or instruction, display the exact message: 'Could not find any medicine in the image you provided'.
End by asking the user what they would like to know about the identified items or general health. But if you find medicine, 
format the identified medicine names in a list within <MED_LIST> tags in your response, comma-separated. For example: <MED_LIST>Aspirin, Ibuprofen</MED_LIST>.


Extracted Text:
"{extracted_text}"

Relevant Lines:
"{'; '.join(state['prescription_details'])}"

Your response:"""

        try: # <--- Inner try block, Indentation level 3
            logger.info("Calling Gemini model for medicine identification...")
            llm_response = gemini_model.generate_content(llm_prompt)
            response_text = llm_response.text.strip()
            logger.info(f"LLM Identification Response: '{response_text[:200]}{'...' if len(response_text) > 200 else ''}'")

            state["assistant_response"] = response_text

            # --- Parse LLM Response for Detected Medicines ---
            # Look for the <MED_LIST>...</MED_LIST> tag
            med_list_match = re.search(r'<MED_LIST>(.*?)</MED_LIST>', response_text, re.IGNORECASE)
            detected_medicines = []
            if med_list_match:
                meds_str = med_list_match.group(1)
                # Split by comma, strip whitespace, filter empty strings
                detected_medicines = [med.strip() for med in meds_str.split(',') if med.strip()]
                logger.info(f"Parsed detected medicines from LLM response: {detected_medicines}")
            else:
                 logger.warning("Could not find <MED_LIST> tag in LLM response.")
                 # Fallback: Try to find capitalized words that look like potential meds, but be very cautious
                 # This fallback is less reliable than prompting the LLM correctly
                 # You might consider removing this fallback for cleaner logic if the tag parsing is sufficient
                 # For now, keeping a refined version: look for words with initial caps, longer than 1 char,
                 # not in a short list of common non-drug words often capitalized.
                 words = re.findall(r'\b[A-Z][a-zA-Z]{1,}\b', response_text) # Words starting with Cap, at least 2 letters
                 common_english_words_caps = {'I', 'You', 'The', 'And', 'But', 'Or', 'Is', 'Are', 'Be', 'To', 'Of', 'In', 'On', 'At', 'For', 'With', 'About', 'From', 'By', 'As', 'It', 'Its', 'Can', 'Will', 'May', 'Please', 'Find', 'Found', 'Here', 'These', 'What', 'Would', 'Like', 'Know', 'About', 'Any', 'Else', 'Prescription', 'Details', 'Extracted', 'Relevant', 'Lines', 'Text', 'Response', 'Summary', 'Identified', 'Medications', 'Names', 'Dosages', 'Instructions', 'Potential', 'Items', 'General', 'Health', 'Note', 'Also', 'See', 'Per', 'Dr', 'Take', 'Use', 'Day', 'Times', 'Mg', 'Ml'}
                 potential_meds_fallback = [word for word in words if word not in common_english_words_caps]
                 detected_medicines = list(set(potential_meds_fallback)) # Use set to remove duplicates
                 if potential_meds_fallback:
                      logger.info(f"Fallback detected potential meds from LLM response text (no tag): {detected_medicines}")

            state["detected_medicines"] = detected_medicines

        except Exception as e: # <--- Inner except block, Indentation level 3
            logger.error(f"Error during LLM interaction or parsing: {str(e)}", exc_info=True)
            # If LLM call/parsing fails, provide an error but keep the extracted text
            state["assistant_response"] = f"Error processing image with AI: {str(e)}. I could still extract text: '{extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}'"
            state["detected_medicines"] = [] # Ensure empty on LLM error
            # No 'return state' here; execution continues *inside* the outer try block

    except Exception as e: # <--- Outer except block, Indentation level 2 (matching outer 'try')
        # This outer except catches errors during image decoding, OCR, or basic text checks
        logger.error(f"Error during image processing (decoding/OCR): {str(e)}", exc_info=True)
        state["assistant_response"] = f"Error processing image: {str(e)}. Please try again with a different image format or a clearer photo."
        state["detected_medicines"] = []
        state["prescription_details"] = []
        state["extracted_text"] = None # Clear text if OCR failed
        # No 'return state' here; execution continues *after* the outer except block

    # This return statement is correctly placed outside both the inner and outer try/except blocks.
    # It will be reached unless an earlier 'return state' within an 'if' block was hit.
    return state # <--- Final return, Indentation level 1 (matching 'def')


# Define the health expert node (Processes text query for general health info)
def health_expert(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Executing health_expert node.")

    if not gemini_model:
        logger.error("Gemini model not initialized.")
        state["assistant_response"] = "AI model not available for health questions."
        return state

    # This node runs for text-only input, so detected_medicines will be empty
    state["detected_medicines"] = []
    state["prescription_details"] = [] # Ensure these are clear for text input path

    # Format history for the prompt
    history_context = ""
    for msg in state.get("history", []):
        role = "User" if msg.role == "user" else "Assistant"
        history_context += f"{role}: {msg.content}\n"

    # Construct the prompt for the language model - focuses on general Q&A
    prompt = f"""You are a helpful medical assistant providing information about medicines and health questions.
Adhere to the following guidelines:
1. Provide accurate, evidence-based information.
2. Always include a disclaimer that information is not medical advice.
3. Be clear, conversational, and empathetic.
4. Never diagnose or recommend treatments or specific dosages.
5. When discussing medications, mention common uses, side effects, and precautions.
6. For specifics like dosage, drug interactions, or treatment plans, recommend consulting a professional.
7. Use simple language.
8. Acknowledge limitations and suggest professional help when uncertain.
9. Advise immediate medical attention for emergencies.

User's question: {state['current_message']}

Previous conversation:
{history_context}

Your response:"""

    try:
        logger.info("Generating Gemini response for health question.")
        gemini_response = gemini_model.generate_content(prompt)
        response = gemini_response.text.strip()

        # Ensure disclaimer is included
        if "not a substitute for professional medical advice" not in response.lower():
            response += "\n\nRemember, this information is not a substitute for professional medical advice. Please consult your healthcare provider for personalized recommendations."
        logger.info(f"Gemini response generated for text query: '{response[:200]}...'")

        state["assistant_response"] = response

    except Exception as e:
        logger.error(f"Error generating Gemini response in health_expert: {str(e)}", exc_info=True)
        state["assistant_response"] = "I'm sorry, I encountered an issue while processing your question. Please try again or ask another question about health or medicine."

    return state

# --- LangGraph Setup ---

def create_graph():
    """Creates and compiles the LangGraph workflow."""
    logger.info("Creating LangGraph workflow.")
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("medicine_extraction", extract_medicines)
    workflow.add_node("health_expert", health_expert)

    # Conditional edge from START: route based on route_step function
    # If image -> medicine_extraction
    # If no image -> health_expert
    workflow.add_conditional_edges(
        START,
        route_step,
        {
            "medicine_extraction": "medicine_extraction",
            "health_expert": "health_expert",
        }
    )

    # Changed edge: After medicine_extraction, END the graph (LLM response is final for this path)
    workflow.add_edge("medicine_extraction", END)

    # Simple edge: After health_expert, END the graph
    workflow.add_edge("health_expert", END)

    compiled_graph = workflow.compile()
    logger.info("LangGraph workflow compiled.")
    return compiled_graph

# Initialize the graph instance
graph = create_graph()


# --- FastAPI Endpoints ---

# Main chat endpoint handling both text and image
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info("Received /chat request")
    logger.info(f"Message: '{request.message[:100]}...'") # Log first 100 chars of message
    logger.info(f"History length: {len(request.history)}")
    logger.info(f"Image received: {request.image is not None}")

    # Initial state for the graph
    state = {
        "current_message": request.message,
        "history": request.history,
        "image": request.image, # Pass the image data directly
        "detected_medicines": [], # Start empty
        "prescription_details": [], # Start empty
        "extracted_text": None, # Start empty
        "assistant_response": None # Start empty
    }

    try:
        # Invoke the graph. The routing logic within the graph handles the flow.
        # The result will contain the final state after the graph runs to END
        result = graph.invoke(state)

        # The final result should contain the assistant_response and detected_medicines (if image path was taken)
        response = result.get("assistant_response", "Sorry, I couldn't process your request.")
        # detected_medicines will be populated by extract_medicines node if image was processed, otherwise remains empty
        medicines = result.get("detected_medicines", [])

        logger.info(f"Chat processing complete. Medicines detected (if image): {medicines}")

        return ChatResponse(response=response, detected_medicines=medicines)

    except Exception as e:
        # Log the full traceback for unexpected errors
        logger.error(f"Unhandled error processing /chat request: {str(e)}", exc_info=True)
        # Provide a generic error message to the user while logging details server-side
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your request.")

# --- Main Execution ---

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # reload=True for development