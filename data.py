import os
import zipfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import clean_text

DATA_DIR = "data"
ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = os.path.join(DATA_DIR, "smsspamcollection.zip")
FILE_PATH = os.path.join(DATA_DIR, "SMSSpamCollection")

def download_and_extract_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if not os.path.exists(FILE_PATH):
        print(f"Downloading dataset from {ZIP_URL}...")
        try:
            response = requests.get(ZIP_URL, timeout=30)
            response.raise_for_status()
            with open(ZIP_PATH, 'wb') as f:
                f.write(response.content)
            
            print("Extracting zip...")
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print("Dataset successfully downloaded and extracted.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Creating a fallback mock dataset for local execution...")
            create_mock_dataset()

def create_mock_dataset():
    # A simple mock dataset for fallback
    mock_data = [
        ("ham", "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."),
        ("ham", "Ok lar... Joking wif u oni..."),
        ("spam", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"),
        ("ham", "U dun say so early hor... U c already then say..."),
        ("ham", "Nah I don't think he goes to usf, he lives around here though"),
        ("spam", "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"),
        ("ham", "Even my brother is not like to speak with me. They treat me like aids patent."),
        ("ham", "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Caller tune"),
        ("spam", "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."),
        ("spam", "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986906"),
    ] * 100 # Duplicate to make it slightly larger for split
    
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        for label, text in mock_data:
            f.write(f"{label}\t{text}\n")

def load_data():
    download_and_extract_data()
    
    print(f"Loading dataset from {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH, sep='\t', names=['label', 'text'], header=None, encoding='utf-8')
    
    # Map label to numerical: spam -> 1, ham -> 0
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Preprocess text
    print("Preprocessing text data...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Drop rows with empty clean text
    df = df[df['clean_text'] != ""]
    
    return df

def get_train_test_data(test_size=0.2, random_state=42):
    df = load_data()
    X = df['clean_text']
    y = df['label_num']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(X_train.head())
