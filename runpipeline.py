from src.data_loader import load_data
from src.feature_engineering import preprocess
from src.train_model import train_model
from src.evaluate_model import evaluate

if __name__ == "__main__":
    df = load_data("data/raw/train.csv")
    df_clean = preprocess(df)
    model = train_model(df_clean)
    evaluate(model, df_clean)
