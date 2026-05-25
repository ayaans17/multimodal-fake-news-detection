import pandas as pd
from sklearn.model_selection import train_test_split

# load dataset
#df = pd.read_csv("../../data/processed/cleaned_dataset.csv")
ev_df = pd.read_csv("..\\Dataset\\processed\\evons\\ev_processed.csv")
#
# ifnd_df = pd.read_csv("..\\Dataset\\processed\\ifnd\\ifnd.csv",encoding="latin-1")
buzz_df = pd.read_csv("..\\Dataset\\processed\\buzzfeed\\bf_processed.csv")

# split
def split_dataset(df,df_name):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['Label'],
        random_state=42
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df['Label'],
        random_state=42
    )

    # save
    train_df.to_csv("..\\Dataset\\processed\\"+df_name + "\\train.csv", index=False)
    val_df.to_csv("..\\Dataset\\processed\\"+df_name + "\\val.csv", index=False)
    test_df.to_csv("..\\Dataset\\processed\\"+df_name + "\\test.csv", index=False)

    print("Dataset split completed.")
split_dataset(ev_df, "evons")


