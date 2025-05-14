# 🩺 MediChat: Prescription Analysis & Health Q\&A Chatbot

MediChat is an AI-powered chatbot system designed to analyze prescription images and answer health-related questions. It leverages FastAPI, EasyOCR, and LangGraph to provide accurate and context-aware responses.

---

## 🚀 Features

* **Prescription Analysis**: Extracts medication names and details from prescription images using EasyOCR.
* **Health Q\&A**: Answers health and medicine-related questions using open-source NLP models.
* **Chat Memory**: Maintains conversation context to enhance response relevance.
* **Mobile-Friendly Interface**: Responsive design suitable for all devices.

---

## 🧠 System Architecture

### Backend

* **FastAPI**: Handles API endpoints for chat, image upload, and prescription analysis.
* **LangGraph**: Manages workflow with custom nodes:

  * **Medicine Extraction Node**: Utilizes EasyOCR for text extraction from images.
  * **Health Expert Node**: Provides medical information and answers using open-source NLP models.
* **Open-Source Models**: Employs free models for OCR and health Q\&A.([Analytics Vidhya][1])

### Frontend

* **React**: Interactive and responsive user interface.
* **Tailwind CSS**: Styling and responsive design.
* **Axios**: Handles API communication.

---

## ⚙️ Setup and Installation

### Backend

1. **Install Python dependencies**:

   
bash
   pip install fastapi uvicorn python-multipart easyocr pillow torch transformers langchain langgraph




2. **Run the backend**:

   
bash
   cd backend
   python main.py




### Frontend

1. **Install Node.js dependencies**:

   
bash
   cd frontend
   npm install




2. **Start the development server**:

   
bash
   npm start




---

## 📁 Project Structure

medichat/
├── backend/
│   ├── main.py             # FastAPI application with LangGraph workflow
│   └── README.md           # Backend setup instructions
└── frontend/
    ├── public/             # Static files
    ├── src/
    │   ├── App.js          # Main React component
    │   ├── index.js        # React entry point
    │   └── index.css       # Global styles with Tailwind
    ├── package.json        # Frontend dependencies
    ├── tailwind.config.js  # Tailwind CSS configuration
    └── README.md           # Frontend setup instructions




---

## 🧪 How It Works

1. **User Input**: User inputs a message or uploads a prescription image.
2. **Backend Processing**:

   * **Medicine Extraction Node**: Extracts medication names from images/text using EasyOCR.
   * **Health Expert Node**: Provides medical information and answers using open-source NLP models.
3. **Response Delivery**: The response is returned to the frontend and displayed to the user.
4. **Context Maintenance**: Chat history is maintained for context in future interactions.

---

## 🛠️ Customization

* **Advanced Medicine Extraction**: Integrate specialized Named Entity Recognition (NER) models for more accurate extraction.
* **Enhanced Health Expert**: Incorporate domain-specific knowledge bases to improve response accuracy.
* **User Authentication**: Implement authentication mechanisms for personalized health records.
* **Data Persistence**: Add database support to store chat history and user data.([GeeksforGeeks][2])

---