from src.train import build_model
from src.preprocess import prepare_dataset

# Load data
train_gen, (X_val, y_val) = prepare_dataset()

# Build and train the model
model = build_model()
model.fit(train_gen, validation_data=(X_val, y_val), epochs=10)

# Save model
model.save("models/cancer_detector.h5")
print("Model training complete. Saved to models/cancer_detector.h5")
