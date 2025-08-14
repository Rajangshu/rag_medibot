# RAG Medibot

A Retrieval-Augmented Generation based chatbot for answering medical questions. This project uses a combination of a large language model and a vector database to provide accurate and context-aware responses to user queries.

## Description

This project implements a medical chatbot that leverages the power of Retrieval-Augmented Generation (RAG). The chatbot is designed to assist users by providing information on a wide range of medical topics. It uses a vector database to store and retrieve relevant medical documents, which are then used to generate responses with a large language model. This approach ensures that the answers are not only contextually relevant but also grounded in factual medical literature.

## Features

* **Retrieval-Augmented Generation (RAG):** The core of the chatbot is the RAG model, which combines a retriever and a generator. The retriever fetches relevant medical documents from a vector database, and the generator uses these documents to create a comprehensive answer.
* **Vector Database:** The project uses a vector database to store embeddings of medical documents. This allows for efficient similarity search and retrieval of relevant information.
* **Streamlit Interface:** The chatbot has a user-friendly web interface built with Streamlit. Users can interact with the chatbot in a conversational manner.
* **Modular and Extendable:** The project is designed to be modular, making it easy to add new data sources, and to experiment with different language models and vector databases.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.8 or higher
* Pip
* Virtualenv (recommended)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Rajangshu/rag_medibot.git](https://github.com/Rajangshu/rag_medibot.git)
    cd rag_medibot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the environment variables:**

    Create a `.env` file in the root directory and add the following:
    ```
    [Your Hugging Face API key]
    ```

### Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser:**

    Navigate to `http://localhost:8501` to interact with the chatbot.

## Technologies Used

* **Python:** The core programming language used for the project.
* **Hugging Face Transformers:** For accessing and using pre-trained language models.
* **FAISS:** For efficient similarity search in the vector database.
* **Streamlit:** For creating the web interface.
* **LangChain:** To chain together the different components of the RAG model.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
