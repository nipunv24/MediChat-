import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FaPaperPlane, FaImage, FaRobot, FaUser, FaSpinner, FaInfo, FaTimes } from 'react-icons/fa';
import { GiMedicines } from 'react-icons/gi';

const API_URL = 'http://localhost:8000';

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [showIntro, setShowIntro] = useState(true);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Adjust textarea height based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(100, textareaRef.current.scrollHeight)}px`;
    }
  }, [input]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle file selection
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check if file is an image
    if (!file.type.match('image.*')) {
      alert('Please select an image file');
      return;
    }

    setImageFile(file);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  // Handle file button click
  const handleFileButtonClick = () => {
    fileInputRef.current.click();
  };

  // Clear image preview and file
  const clearImage = () => {
    setImagePreview(null);
    setImageFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Close intro panel
  const closeIntro = () => {
    setShowIntro(false);
  };

  // Handle send message
  const handleSend = async () => {
    if (!input.trim() && !imageFile) return;

    const userMessage = input.trim();
    setInput('');

    // Add user message to chat
    const newMessages = [
      ...messages,
      { role: 'user', content: userMessage, timestamp: new Date() }
    ];
    
    // If there's an image, add it to the message
    let imageData = null;
    if (imageFile) {
      const reader = new FileReader();
      imageData = await new Promise((resolve) => {
        reader.onload = (e) => resolve(e.target.result);
        reader.readAsDataURL(imageFile);
      });
      
      // Add image preview to messages
      newMessages.push({ 
        role: 'user', 
        content: 'Uploaded prescription image', 
        image: imagePreview,
        timestamp: new Date() 
      });
    }
    
    setMessages(newMessages);
    setIsLoading(true);
    
    // Add loading message immediately
    const updatedMessages = [
      ...newMessages,
      { role: 'assistant', isLoading: true, timestamp: new Date() }
    ];
    setMessages(updatedMessages);
    
    clearImage();
    
    // Close intro panel if it's still open
    if (showIntro) {
      setShowIntro(false);
    }

    try {
      // Prepare history
      const history = messages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // Send request to backend
      const response = await axios.post(`${API_URL}/chat`, {
        message: userMessage,
        history: history,
        image: imageData
      });

      // Replace loading message with actual response
      updatedMessages.pop(); // Remove loading message
      
      // Add bot response to chat
      setMessages([
        ...updatedMessages,
        { 
          role: 'assistant', 
          content: response.data.response,
          medicines: response.data.detected_medicines,
          timestamp: new Date() 
        }
      ]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Replace loading message with error
      updatedMessages.pop(); // Remove loading message
      
      // Add error message
      setMessages([
        ...updatedMessages,
        { 
          role: 'assistant', 
          content: 'Sorry, there was an error processing your request. Please check if the backend server is running and try again.',
          timestamp: new Date(),
          isError: true
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Format the message content with medicines highlighted
  const formatMessage = (message) => {
    if (!message.medicines || message.medicines.length === 0) {
      return message.content;
    }

    // Simple formatting for highlighting medicines
    let formattedContent = message.content;
    message.medicines.forEach(medicine => {
      // Case-insensitive replacement with highlighting
      const regex = new RegExp(`\\b${medicine}\\b`, 'gi');
      formattedContent = formattedContent.replace(regex, `<span class="font-bold text-green-600">${medicine}</span>`);
    });

    return (
      <div dangerouslySetInnerHTML={{ __html: formattedContent }} />
    );
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center">
            <GiMedicines className="text-2xl mr-2" />
            <h1 className="text-xl font-bold">MediChat</h1>
          </div>
          <div className="text-sm">AI Health Assistant</div>
        </div>
      </header>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 relative">
        <div className="container mx-auto max-w-3xl">
          {/* Intro panel */}
          {showIntro && (
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6 relative">
              <button 
                onClick={closeIntro}
                className="absolute top-2 right-2 text-gray-500 hover:text-gray-800"
              >
                <FaTimes />
              </button>
              <div className="flex items-center mb-4">
                <FaInfo className="text-blue-500 mr-2 text-xl" />
                <h2 className="text-lg font-semibold">Welcome to MediChat!</h2>
              </div>
              <p className="mb-4">
                I'm your AI health assistant. Here's how I can help you:
              </p>
              <ul className="list-disc pl-5 mb-4 space-y-2">
                <li>Ask questions about medications, their uses, and side effects</li>
                <li>Upload a prescription image for analysis</li>
                <li>Get general health information and advice</li>
              </ul>
              <p className="text-sm text-gray-600 italic">
                Remember: All information provided is educational and not a substitute for professional medical advice.
              </p>
            </div>
          )}

          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
              <GiMedicines className="text-5xl mb-4 text-blue-500" />
              <p className="text-center text-lg">
                Ask questions about medications or upload a prescription image.
              </p>
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-lg">
                <div 
                  className="bg-white p-4 rounded-lg shadow-md text-center cursor-pointer hover:bg-blue-50"
                  onClick={() => setInput("What is amoxicillin used for?")}
                >
                  "What is amoxicillin used for?"
                </div>
                <div 
                  className="bg-white p-4 rounded-lg shadow-md text-center cursor-pointer hover:bg-blue-50"
                  onClick={() => setInput("Side effects of ibuprofen?")}
                >
                  "Side effects of ibuprofen?"
                </div>
                <div 
                  className="bg-white p-4 rounded-lg shadow-md text-center cursor-pointer hover:bg-blue-50"
                  onClick={() => setInput("Can I take aspirin with lisinopril?")}
                >
                  "Can I take aspirin with lisinopril?"
                </div>
                <div 
                  className="bg-white p-4 rounded-lg shadow-md text-center cursor-pointer hover:bg-blue-50"
                  onClick={handleFileButtonClick}
                >
                  "Upload a prescription image"
                </div>
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <div 
                key={index} 
                className={`mb-4 flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div 
                  className={`rounded-lg p-3 max-w-3/4 shadow ${
                    message.role === 'user' 
                      ? 'bg-blue-500 text-white' 
                      : message.isError 
                        ? 'bg-red-50 text-red-800 border border-red-200'
                        : 'bg-white text-gray-800'
                  }`}
                >
                  <div className="flex items-center mb-1">
                    {message.role === 'assistant' ? (
                      <FaRobot className="mr-2" />
                    ) : (
                      <FaUser className="mr-2" />
                    )}
                    <span className="font-bold">
                      {message.role === 'assistant' ? 'MediChat' : 'You'}
                    </span>
                  </div>
                  
                  {message.isLoading ? (
                    <div className="flex items-center py-2">
                      <FaSpinner className="animate-spin mr-2 text-blue-500" />
                      <span className="text-gray-600">Generating response...</span>
                    </div>
                  ) : message.image ? (
                    <div className="mt-2">
                      <img 
                        src={message.image} 
                        alt="Prescription" 
                        className="max-w-xs max-h-48 rounded-lg object-contain" 
                      />
                    </div>
                  ) : (
                    message.role === 'assistant' ? formatMessage(message) : message.content
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Image preview */}
      {imagePreview && (
        <div className="bg-gray-200 p-2">
          <div className="container mx-auto max-w-3xl flex items-center">
            <div className="w-16 h-16 bg-gray-300 rounded overflow-hidden mr-2">
              <img 
                src={imagePreview} 
                alt="Preview" 
                className="w-full h-full object-cover" 
              />
            </div>
            <div className="flex-1">
              <p className="text-sm truncate">Prescription image ready to send</p>
            </div>
            <button 
              onClick={clearImage}
              className="text-red-500 px-2 py-1 rounded hover:bg-red-100"
            >
              Remove
            </button>
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="bg-white border-t border-gray-300 p-4">
        <div className="container mx-auto max-w-3xl">
          <div className="flex items-center">
            <button
              onClick={handleFileButtonClick}
              className="flex-shrink-0 p-3 rounded-full text-blue-500 hover:bg-blue-100"
              title="Upload prescription image"
            >
              <FaImage />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
            <div className="flex-1 border border-gray-300 rounded-full overflow-hidden flex mx-2">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about medications or health questions..."
                className="flex-1 py-2 px-4 focus:outline-none resize-none"
                rows="1"
              />
            </div>
            <button
              onClick={handleSend}
              disabled={isLoading || (!input.trim() && !imageFile)}
              className={`flex-shrink-0 p-3 rounded-full ${
                isLoading || (!input.trim() && !imageFile)
                  ? 'text-gray-400 bg-gray-100'
                  : 'text-white bg-blue-500 hover:bg-blue-600'
              }`}
            >
              {isLoading ? <FaSpinner className="animate-spin" /> : <FaPaperPlane />}
            </button>
          </div>
          <div className="text-xs text-gray-500 text-center mt-2">
            MediChat provides educational information only, not medical advice.
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;