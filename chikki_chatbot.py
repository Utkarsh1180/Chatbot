import json
import random
import numpy as np
import tkinter as tk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle

# Load model and data
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')

# Load intents.json with correct encoding
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Preprocessing functions
def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, something went wrong."

# GUI using tkinter
def chatbot_gui():
    def send():
        user_input = entry_box.get("1.0", "end-1c").strip()
        entry_box.delete("0.0", "end")

        if user_input:
            chat_log.config(state=tk.NORMAL)
            chat_log.insert(tk.END, "You: " + user_input + '\n\n')
            response = get_response(predict_class(user_input), intents)
            chat_log.insert(tk.END, "Chikki: " + response + '\n\n')
            chat_log.config(state=tk.DISABLED)
            chat_log.yview(tk.END)

    # Create window
    root = tk.Tk()
    root.title("Chikki - AI Chatbot")
    root.geometry("400x500")
    root.resizable(False, False)

    # Chat log
    chat_log = tk.Text(root, bd=1, bg="white", height="8", width="50", font="Arial")
    chat_log.config(state=tk.DISABLED)

    scroll_bar = tk.Scrollbar(root, command=chat_log.yview)
    chat_log['yscrollcommand'] = scroll_bar.set

    # Entry box
    entry_box = tk.Text(root, bd=0, bg="white", width="29", height="4", font="Arial")
    send_button = tk.Button(root, text="Send", width="12", height=2, bd=0, bg="#4CAF50", fg='white', command=send)

    # Layout
    chat_log.place(x=6, y=6, height=386, width=370)
    scroll_bar.place(x=376, y=6, height=386)
    entry_box.place(x=6, y=401, height=90, width=265)
    send_button.place(x=275, y=401, height=90)

    root.mainloop()

# Run the chatbot
chatbot_gui()
