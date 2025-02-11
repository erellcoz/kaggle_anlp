"""Module pour les constantes du projet."""

# PATHS
TRAINING_DATA_PATH = "data/train_submission.csv"
TEST_DATA_PATH = "data/test_without_labels.csv"

# COLUMN NAMES
LABEL_COLUMN = "Label"
TEXT_COLUMN = "Text"
EMBEDDING_COLUMN = "Embedding"

# EMBEDDING
EMBEDDING_SIZE = 512

# ALPHABETS
# Définition des plages Unicode pour différents alphabets
ALPHABETS = {
    "Latin": (0x0041, 0x007A),
    "Cyrillique": (0x0400, 0x04FF),
    "Arabe": (0x0600, 0x06FF),
    "Hébreu": (0x0590, 0x05FF),
    "Chinois": (0x4E00, 0x9FFF),
    "Hiragana": (0x3040, 0x309F),
    "Katana": (0x30A0, 0x30FF),
    "Coréen": (0xAC00, 0xD7AF),
    "Grec": (0x0370, 0x03FF),
    "Gujarati": (0x0A80, 0x0AFF),
    "Devanagari": (0x0900, 0x097F),
    "Thaï": (0x0E00, 0x0E7F),
    "Géorgien": (0x10A0, 0x10FF),
}
