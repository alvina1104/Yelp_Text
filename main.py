from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from googletrans import Translator


classes = {
    0: '1 star',
    1: '2 stars',
    2: '3 stars',
    3: '4 stars',
    4: '5 stars',
}

class ChrckYelp(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()

    self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
    self.lstm = nn.LSTM(64, 128, batch_first=True)
    self.lin = nn.Linear(128, 5)

  def forward(self, text):
    text = self.emb(text)
    _, (hid, _) = self.lstm(text)
    return self.lin(hid[-1])


vocab = torch.load('yelp_vocab.pth', weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChrckYelp(len(vocab))
model.load_state_dict(torch.load('yelp.pth', map_location=device))
model.to(device)
model.eval()

yelp_app = FastAPI()

tokenizer = get_tokenizer('basic_english')

def change_text(words):
  return [vocab[i] for i in tokenizer(words)]

class TextSchema(BaseModel):
    word: str

translator = Translator()

@yelp_app.post('/predict')
async def predict_text(text: TextSchema):
    translated = await translator.translate(text.word, dest='en')
    translate_text = translated.text

    num_text = torch.tensor(change_text(translate_text), dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(num_text)
        result = torch.argmax(prediction, dim=1).item()

    return {
        'translation': translate_text,
        'prediction': classes[result]
    }

if __name__ == '__main__':
    uvicorn.run(yelp_app, host="127.0.0.1", port=8000)
