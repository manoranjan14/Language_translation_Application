import os
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from transformers import MarianTokenizer, TFMarianMTModel

class TranslationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Translation App")
        self.geometry("600x400")
        self.create_widgets()
        self.load_models()

    def create_widgets(self):
        label = tk.Label(self, text="Enter English text to translate:", font=("Arial", 14))
        label.pack(pady=20)

        self.text_input = ScrolledText(self, wrap=tk.WORD, width=60, height=10)
        self.text_input.pack(pady=20)

        button_translate_french = ttk.Button(self, text="Translate to French", command=self.translate_to_french)
        button_translate_french.pack(pady=10)

        button_translate_urdu = ttk.Button(self, text="Translate to Urdu", command=self.translate_to_urdu)
        button_translate_urdu.pack(pady=10)

        button_exit = ttk.Button(self, text="Exit", command=self.destroy)
        button_exit.pack(pady=10)

        self.result_label = tk.Label(self, text="", font=("Arial", 12), wraplength=500)
        self.result_label.pack(pady=20)

    def load_models(self):
        # Directory where models will be saved
        model_directory = "C:\\translator_app\\models"
        os.makedirs(model_directory, exist_ok=True)

        # Load and save the pre-trained models and tokenizers
        self.tokenizer_en_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.tokenizer_en_fr.save_pretrained(model_directory)

        self.model_en_fr = TFMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.model_en_fr.save_pretrained(model_directory)

        self.tokenizer_en_ur = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
        self.tokenizer_en_ur.save_pretrained(model_directory)

        self.model_en_ur = TFMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ur")
        self.model_en_ur.save_pretrained(model_directory)

    def translate_to_french(self):
        user_input = self.text_input.get("1.0", tk.END).strip()
        if user_input:
            translated_text = self.translate(user_input, self.tokenizer_en_fr, self.model_en_fr, "French")
            self.result_label.config(text=f"Translated to French: {translated_text}")
        else:
            messagebox.showwarning("Input Error", "Please enter text for translation.")

    def translate_to_urdu(self):
        user_input = self.text_input.get("1.0", tk.END).strip()
        if user_input:
            translated_text = self.translate(user_input, self.tokenizer_en_ur, self.model_en_ur, "Urdu")
            self.result_label.config(text=f"Translated to Urdu: {translated_text}")
        else:
            messagebox.showwarning("Input Error", "Please enter text for translation.")

    def translate(self, input_text, tokenizer, model, target_language):
        # Tokenize the input text
        inputs = tokenizer([input_text], return_tensors="tf", truncation=True, padding=True)

        # Perform translation
        translated = model.generate(**inputs)

        # Decode the translated tokens and convert to text
        translated_text = tokenizer.decode(translated.numpy()[0], skip_special_tokens=True)

        return translated_text

if __name__ == "__main__":
    app = TranslationApp()
    app.mainloop()
