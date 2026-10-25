import torch

history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
for train_loss, validation_loss, validation_accuracy in [
    (1.8, 1.7, 0.42), (1.1, 1.2, 0.66), (0.6, 1.4, 0.61)
]:
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["validation_accuracy"].append(validation_accuracy)
assert len(history["train_loss"]) == 3
assert history["train_loss"][-1] < history["train_loss"][0]
assert history["validation_loss"][-1] > history["validation_loss"][-2]
print("history recorded; final epoch shows a train/validation gap")
