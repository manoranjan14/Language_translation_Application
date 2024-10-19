# Language Translation Application

This project is a desktop translation application built with Python's Tkinter for the graphical user interface (GUI). The app allows users to translate English text into French and Urdu using pre-trained MarianMT models from Hugging Face's `transformers` library. The application provides real-time translation with models stored locally for improved performance.

## Features

- **Simple User Interface**: A user-friendly GUI where users can input English text and translate it into French or Urdu.
- **Pre-trained MarianMT Models**: Leverages high-quality MarianMT models for English-to-French and English-to-Urdu translations.
- **Real-time Translation**: Immediate translation upon text input.
- **Local Model Storage**: Models are saved locally to avoid repeated downloads, enhancing speed and performance.

## Technologies Used

- **Python**: Core programming language.
- **Tkinter**: For building the GUI.
- **Hugging Face Transformers**: For using pre-trained MarianMT models.
- **TensorFlow**: Backend for running the MarianMT models.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/manoranjan14/Language_translation_Application.git
cd Language_translation_Application
```

### Step 2: Install Required Dependencies

Ensure you have Python 3.7+ installed, then run:

```bash
pip install -r requirements.txt
```

The required dependencies are listed in `requirements.txt`.

### Step 3: Run the Application

```bash
python final.py
```

The app window will launch, allowing you to input English text and translate it into French or Urdu.

## Project Structure

```plaintext
Language_translation_Application/
├── final.py               # Main Python script to run the app
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── LICENSE                # License file
```

## Future Enhancements

- **Add More Languages**: Support for additional languages by including more MarianMT models.
- **Optimize Performance**: Implement threading for background model loading to keep the UI responsive.
- **Improve Error Handling**: Enhance error messages for invalid inputs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
