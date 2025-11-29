import pandas as pd

class DataLoader:
    """
    Loads datasets from external files.
    """

    @staticmethod
    def load_movies(path: str) -> pd.DataFrame:
        """
        Loads movies_metadata.csv from given path.
        """
        try:
            df = pd.read_csv(path, low_memory=False)
            print(f"Data loaded successfully. Rows: {len(df)}")
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
