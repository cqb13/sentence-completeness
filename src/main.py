import torch
import torch.nn as nn
import torch.optim as optim
import threading

from data import training_data

class SentenceCompletion(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SentenceCompletion, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

input_size = 40 # the number of features for each word
hidden_size = 300
output_size = 2 # 1 for complete sentence, 0 for incomplete sentence

learning_rate = 0.01
epochs = 1000

# Instantiate the neural network
model = SentenceCompletion(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Determine the maximum sentence length
max_length = max([len(sentence.split()) for sentence, _ in training_data])

# Convert the training data into tensors
inputs = []
targets = []
for sentence, target in training_data:
    sentence_vec = torch.zeros(max_length, input_size)
    for i, word in enumerate(sentence.split()):
        word_vec = torch.randn(input_size)
        sentence_vec[i] = word_vec
    inputs.append(sentence_vec.unsqueeze(0))
    targets.append(target)
inputs = torch.cat(inputs, dim=0)
targets = torch.tensor(targets, dtype=torch.long)

# Define a function to train the neural network
def train(model, inputs, targets, criterion, optimizer, epoch):
    outputs = model(inputs)

    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def predict(model, sentence, input_size):
    # Convert the sentence to a tensor
    sentence_vec = torch.zeros(len(sentence), input_size)
    for i, word in enumerate(sentence.split()):
        word_vec = torch.randn(input_size)
        sentence_vec[i] = word_vec

    output = model(sentence_vec.unsqueeze(0))
    predicted_target = output.argmax(dim=1).item()

    if predicted_target == 1:
        print(f"'{sentence}' \n (complete sentence)")
    else:
        print(f"'{sentence}' \n (incomplete sentence)")

# Train the neural network
for epoch in range(epochs):
    thread = threading.Thread(target=train, args=(model, inputs, targets, criterion, optimizer, epoch))
    thread.start()
    thread.join()

#custom sentence
sentence = "The quick red fox jumps over the lazy brown dog"
predict(model, sentence, input_size)
