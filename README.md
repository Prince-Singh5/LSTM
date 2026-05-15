# Next Word Prediction using LSTM

This project implements a **Next Word Prediction model** using **TensorFlow/Keras** with an LSTM-based architecture.  
The model is trained on text data (e.g., Shakespeare's *Hamlet*) to predict the next word in a sequence.

---

## 📌 Project Overview
- Preprocesses text data into sequences of tokens.
- Splits sequences into predictors (`x`) and labels (`y`).
- Trains an LSTM neural network to learn word dependencies.
- Uses the trained model to predict the next word given a sequence.

---

## 🛠 Model Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation="softmax"))
