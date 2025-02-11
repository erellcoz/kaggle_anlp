"""Module afin de créer les embeddings des textes."""

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from constants import EMBEDDING_COLUMN, EMBEDDING_SIZE, TEXT_COLUMN


def get_df_embeddings(df: pd.DataFrame):
    """Créer les embeddings des textes du DataFrame et les sauvegarde
    dans un fichier CSV.

    Args:
        df (pd.DataFrame): DataFrame contenant les textes.

    """
    # à changer pour tester d'autres modèles

    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    # Suivi de la progression avec tqdm
    with tqdm(total=len(df), desc="Enbedding en cours", unit="embedding") as pbar:
        for index, row in df.iterrows():
            # Ajouter la colonne des embeddings
            df.at[index, EMBEDDING_COLUMN] = model.encode(row[TEXT_COLUMN])
            pbar.update(1)

    df.to_csv("data/train_data_with_embedding.csv", index=False)

    for i in range(EMBEDDING_SIZE):
        df[f"embedding_{i}"] = df[EMBEDDING_COLUMN].apply(lambda x: x[i])

    df.to_csv("data/train_data_with_embedding_per_column.csv", index=False)


if __name__ == "__main__":
    # Charger les données
    df = pd.read_csv("data/train_submission.csv")

    # Créer les embeddings
    get_df_embeddings(df)
