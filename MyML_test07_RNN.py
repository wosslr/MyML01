from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN
from passage.utils import save, load

tokenizer = Tokenizer()
train_tokens = tokenizer.fit_transform(train_text)

layers = [
    Embedding(size=128, n_features=tokenizer.n_features),
    GatedRecurrent(size=128),
    Dense(size=1, activation='sigmoid')
]

model = RNN(layers=layers, cost='BinaryCrossEntropy')
model.fit(train_tokens, train_labels)

model.predict(tokenizer.transform(test_text))
save(model, 'save_test.pkl')
model = load('save_test.pkl')




