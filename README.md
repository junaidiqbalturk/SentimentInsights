# Sentiment Insight

This project is a sentiment analysis web application built with Flask and trained on the IMDB movie reviews dataset. It allows users to input text and get predictions about the sentiment (positive or negative) of the text. The application features a basic but functional user interface for interacting with the sentiment analysis model.

## Features

- **Sentiment Analysis:** Classifies text input as positive or negative.
- **Flask Web Application:** Provides a user-friendly interface to interact with the sentiment analysis model.
- **Responsive Design:** Basic front-end design that can be further enhanced for responsiveness and modern aesthetics.

## Getting Started

To run this project locally, follow these steps:

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repository.git

2. **Navigate to the Project Directory**

    ```bash
   cd your-repository

3. **Create a Virtual Environment**

    ```bash
   python3 -m venv venv

4. **Activate the Virtual Environment**

    **On Windows:**
    ```bash
    venv\Scripts\activate
    **On macOS/Linux:**
    source venv/bin/activate

5. **Install the Dependencies**

      ```bash
   pip install -r requirements.txt

6. **Download NLTK Data**

      ```bash
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')

7. **Run the Application**

   ```bash
   python run.py

## Usage
   1. Open the web application in your browser.
   2. Enter the text you want to analyze in the input field.
   3. Click the "Analyze" button to get the sentiment prediction.
   4. View the results on the same page.


## Contribution
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgement
1. NLTK for natural language processing tools.
2. Flask for the web framework.
3. IMDB Dataset for the sentiment analysis data.