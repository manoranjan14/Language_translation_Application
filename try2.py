import os
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from transformers import MarianTokenizer, TFMarianMTModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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

        button_evaluate = ttk.Button(self, text="Evaluate Accuracy", command=self.evaluate_accuracy)
        button_evaluate.pack(pady=10)

        button_exit = ttk.Button(self, text="Exit", command=self.destroy)
        button_exit.pack(pady=10)

        self.result_label = tk.Label(self, text="", font=("Arial", 12), wraplength=500)
        self.result_label.pack(pady=20)

    def load_models(self):
        model_directory = "C:\\translator_app\\models"
        os.makedirs(model_directory, exist_ok=True)

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
            translated_text = self.translate(user_input, self.tokenizer_en_fr, self.model_en_fr)
            self.result_label.config(text=f"Translated to French: {translated_text}")
        else:
            messagebox.showwarning("Input Error", "Please enter text for translation.")

    def translate_to_urdu(self):
        user_input = self.text_input.get("1.0", tk.END).strip()
        if user_input:
            translated_text = self.translate(user_input, self.tokenizer_en_ur, self.model_en_ur)
            self.result_label.config(text=f"Translated to Urdu: {translated_text}")
        else:
            messagebox.showwarning("Input Error", "Please enter text for translation.")

    def translate(self, input_text, tokenizer, model):
        inputs = tokenizer([input_text], return_tensors="tf", truncation=True, padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated.numpy()[0], skip_special_tokens=True)
        return translated_text

    def evaluate_accuracy(self):
        test_sentences = ["Hello, how are you?", "What is your name?", "I love programming.", "This is a test sentence."]
        reference_translations_fr = ["Bonjour, comment ça va?", "Quel est ton nom?", "J'aime programmer.", "Ceci est une phrase de test."]
        reference_translations_ur = ["ہیلو، آپ کیسے ہیں؟", "آپ کا نام کیا ہے؟", "مجھے پروگرامنگ پسند ہے۔", "یہ ایک ٹیسٹ جملہ ہے۔"]

        bleu_scores_fr = self.calculate_bleu_scores(test_sentences, reference_translations_fr, self.tokenizer_en_fr, self.model_en_fr)
        bleu_scores_ur = self.calculate_bleu_scores(test_sentences, reference_translations_ur, self.tokenizer_en_ur, self.model_en_ur)

        avg_bleu_fr = sum(bleu_scores_fr) / len(bleu_scores_fr)
        avg_bleu_ur = sum(bleu_scores_ur) / len(bleu_scores_ur)

        result_text = (f"Average BLEU score for French translations: {avg_bleu_fr:.2f}\n"
                       f"Average BLEU score for Urdu translations: {avg_bleu_ur:.2f}")
        messagebox.showinfo("Evaluation Results", result_text)

    def calculate_bleu_scores(self, sentences, references, tokenizer, model):
        bleu_scores = []
        smoothing_function = SmoothingFunction().method1
        for sentence, reference in zip(sentences, references):
            translation = self.translate(sentence, tokenizer, model)
            reference_tokens = reference.split()
            translation_tokens = translation.split()
            bleu_score = sentence_bleu([reference_tokens], translation_tokens, smoothing_function=smoothing_function)
            bleu_scores.append(bleu_score)
        return bleu_scores

if __name__ == "__main__":
    app = TranslationApp()
    app.mainloop()
