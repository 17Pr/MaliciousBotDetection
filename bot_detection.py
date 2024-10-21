import pandas as pd
from learning_automata import LearningAutomata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
def load_data(file_path):
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {len(data)} records.")
    return data

# Preprocess the dataset
def preprocess_data(data):
    print("Preprocessing dataset...")
    # Convert categorical features to numerical values
    le_ip = LabelEncoder()
    data['ip_address'] = le_ip.fit_transform(data['ip_address'])
    
    le_agent = LabelEncoder()
    data['user_agent'] = le_agent.fit_transform(data['user_agent'])
    
    print("Preprocessing completed.")
    return data

# Training function
def train_model(data):
    print("Starting training...")
    features = data[['ip_address', 'user_agent', 'request_rate']]
    target = data['is_malicious']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Initialize Learning Automata with 2 actions: normal or malicious bot
    la = LearningAutomata(num_actions=2)
    
    # Simulate learning process
    for index in range(len(X_train)):  # Use range(len(X_train)) to avoid index misalignment
        action = la.select_action()  # Choose action (0 = normal, 1 = malicious)
        actual_label = y_train.iloc[index]
        
        # Reward if the action matches the actual label, otherwise penalize
        reward = 1 if action == actual_label else 0
        la.update(action, reward)
    
    print("Training completed.")
    return la, X_test, y_test

# Test function
def test_model(la, X_test, y_test):
    print("Starting testing...")
    correct_predictions = 0
    total_predictions = len(X_test)
    
    for index in range(len(X_test)):  # Use range(len(X_test)) for iteration
        action = la.select_action()  # Choose action
        actual_label = y_test.iloc[index]
        
        if action == actual_label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Load and preprocess the data
    data = load_data('data/bot_data.csv')
    data = preprocess_data(data)
    
    # Train the model
    la_model, X_test, y_test = train_model(data)
    
    # Test the model
    test_model(la_model, X_test, y_test)

