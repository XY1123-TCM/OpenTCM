# OpenTCM

OpenTCM is a web application for Traditional Chinese Medicine (TCM) intelligent question answering, built upon a knowledge graph and a Large Language Model (LLM, using Kimi as an example). It aims to combine structured TCM knowledge with the understanding and generation capabilities of LLMs to provide users with well-sourced, comprehensive, and easy-to-understand TCM information. This project supports streaming responses to enhance user interaction experience.

Our paper "OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis" is accepted by BIGCOM25.

The code of OpenTCM will be released by this conference.

Due to privacy and copyright reasons, the dataset is not available. You can download it yourself.

## Technical Architecture

The project follows a front-end/back-end architecture:

* **Backend (Python & Flask)**:
    * `Flask`: Serves as the web framework, providing API endpoints.
    * `tcm_rag_system.py`: Contains the core logic:
        * `TCMKnowledgeGraph`: Construction, management, and querying of the knowledge graph.
        * `GraphRAG`: Implements the GraphRAG pipeline, including interaction with the Kimi API, prompt construction, knowledge retrieval, and answer synthesis.
        * `TCMGraphRAGApp`: Encapsulates the application logic for API calls.
    * `requests`: Communicates with external LLM APIs (e.g., Moonshot Kimi).
    * `python-dotenv`: Manages environment variables (like API keys).
    * `Flask-CORS`: Handles Cross-Origin Resource Sharing.
* **Frontend (HTML, CSS, Vanilla JavaScript)**:
    * Built with pure HTML/CSS/JS for the user interface.
    * Uses the `Workspace` API for asynchronous communication with the backend.
    * Implements streaming reception and display of responses from the backend.

## Technology Stack

* **Backend**: Python 3.x, Flask, Pandas, NetworkX, Requests
* **Frontend**: HTML5, CSS3, JavaScript (ES6+)
* **LLM API**: Moonshot (Kimi) API (can be replaced with other compatible APIs)
* **Environment Management**: python-dotenv

## Project Structure


├── /data       
│   ├── TCMKG.py      
│   └── data.csv    #add your dataset here
├── app.py               
├── GraphRAG.py     
├── .env                 
├── requirements.txt      
├── /templates
│   ├── welcome.html      
│   └── chat.html         
└── /static
├── /css
│   └── style.css     
└── /images
└── opentcm_logo.png
└── readme.md





## How to use
1. Create conda env     # python 3.10 or above is recommended
2. Use TCMKG.py to create your dataset
3. Run app.py
4. Open your web browser.
Navigate to http://127.0.0.1:8000/ or http://localhost:8000/.
You will see the welcome page. Click the "Start Chat" button to go to the chat interface.
Type your TCM-related question into the input box and click "Send" or press Enter.
The system will process your query and display the answer progressively on the interface via streaming output.
