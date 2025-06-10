import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import os

# Define constants for filenames
TRAINING_DATA_FILE = 'petr4_treinamento.csv'
MODEL_FILE = 'model.h5'
FEATURE_SCALER_FILE = 'feature_scaler.pkl'
TARGET_SCALER_FILE = 'target_scaler.pkl'

def train_model():
    # Load and preprocess data
    if not os.path.exists(TRAINING_DATA_FILE):
        print(f"Error: Training data file '{TRAINING_DATA_FILE}' not found.")
        return

    try:
        base = pd.read_csv(TRAINING_DATA_FILE)
        base = base.dropna()
        base_treinamento_features = base.iloc[:, 1:7].values
        base_treinamento_target = base.iloc[:, 1:2].values # Open price

        # Initialize and fit scalers
        feature_scaler = MinMaxScaler(feature_range=(0,1))
        scaled_features = feature_scaler.fit_transform(base_treinamento_features)

        target_scaler = MinMaxScaler(feature_range=(0,1))
        scaled_target = target_scaler.fit_transform(base_treinamento_target)

        # Prepare sequences for LSTM
        previsores = []
        preco_real = []
        for i in range(90, len(scaled_features)):
            previsores.append(scaled_features[i-90:i, 0:6])
            preco_real.append(scaled_target[i, 0])

        previsores, preco_real = np.array(previsores), np.array(preco_real)
        # Reshape previsores: [samples, time steps, features]
        # The number of features is 6 (Open, High, Low, Close, Adj Close, Volume)
        previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 6))

        # Build, compile, and train the LSTM model
        regressor = Sequential()
        regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 6)))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.3))

        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.3))

        regressor.add(Dense(units=1, activation='sigmoid'))

        regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        # Define callbacks
        es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
        rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
        mcp = ModelCheckpoint(filepath=MODEL_FILE, monitor='loss', save_best_only=True, verbose=1)

        # Fit the model
        print("Starting model training...")
        regressor.fit(previsores, preco_real, epochs=5, batch_size=32, callbacks=[es, rlr, mcp])
        print("Model training finished.")

        # Save the scalers
        joblib.dump(feature_scaler, FEATURE_SCALER_FILE)
        joblib.dump(target_scaler, TARGET_SCALER_FILE)

        # Verify files are saved
        model_saved = os.path.exists(MODEL_FILE)
        feature_scaler_saved = os.path.exists(FEATURE_SCALER_FILE)
        target_scaler_saved = os.path.exists(TARGET_SCALER_FILE)

        if model_saved and feature_scaler_saved and target_scaler_saved:
            print("Training complete. Model and scalers saved.")
            print(f"Model saved as: {MODEL_FILE}")
            print(f"Feature scaler saved as: {FEATURE_SCALER_FILE}")
            print(f"Target scaler saved as: {TARGET_SCALER_FILE}")
        else:
            print("Error: Some files were not saved.")
            if not model_saved:
                print(f"- {MODEL_FILE} not found.")
            if not feature_scaler_saved:
                print(f"- {FEATURE_SCALER_FILE} not found.")
            if not target_scaler_saved:
                print(f"- {TARGET_SCALER_FILE} not found.")

    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    train_model()
    # List created files
    print("\nFiles in current directory:")
    for item in os.listdir("."):
        if item == MODEL_FILE or item == FEATURE_SCALER_FILE or item == TARGET_SCALER_FILE:
            print(item)
