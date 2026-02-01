# ezyZip Health - Disease Prediction & AI Chatbot 🏥✨

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.2-000000?style=for-the-badge&logo=flask&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

A modern, AI-powered healthcare application designed to provide instant disease predictions and medical guidance. Using machine learning models, the system analyzes user symptoms to predict potential health conditions and offers personalized drug recommendations. It also features a voice-enabled AI medical assistant for interactive queries.

---

## 🚀 Features

- **Intelligence-Based Prediction**: Utilizes Random Forest and Ensemble models for high-accuracy disease prediction from symptom inputs.
- **Personalized Recommendations**: Provides specific drug suggestions and lifestyle advice for predicted conditions.
- **Interactive AI Chatbot**: A voice-controlled assistant to answer medical questions and provide health tips.
- **Stunning UI/UX**: 
  - **Vibrant Day Mode**: High-saturation, color-graded medical doodle background.
  - **Optimized Dark Mode**: Premium, low-eye-strain theme with persistent state.
- **Glassmorphic Design**: Modern, responsive interface with smooth animations and transitions.

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-Learn, Joblib
- **Frontend**: Vanilla HTML5, CSS3, JavaScript
- **Styling**: Custom CSS (Glassmorphism), Font Awesome Icons, Google Fonts (Inter)
- **Interactive Components**: Choices.js (Dynamic Dropdowns)

---

## 📂 Project Structure

```text
├── app.py                  # Main Flask application & routes
├── requirements.txt        # Project dependencies
├── ensemble_model.pkl      # Machine learning model (Ensemble)
├── medical_rf.pkl          # Machine learning model (Random Forest)
├── static/
│   └── images/
│       └── medical_bg.png  # Adaptive medical background image
└── templates/
    ├── index.html          # Main landing & prediction page
    ├── recommendations.html # Detailed results & drug suggestions
    ├── chat.html           # AI Medical Assistant interface
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8 or higher installed on your system.

### Steps to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Akshat5249/diseaseprediction.git
   cd diseaseprediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the App**
   Open your browser and navigate to:
   ```text
   http://127.0.0.1:5001
   ```

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🌟 Acknowledgments
- Machine Learning models trained on standardized healthcare datasets.
- UI inspiration from modern medical dashboards and glassmorphic aesthetics.

---
*Disclaimer: This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.*
