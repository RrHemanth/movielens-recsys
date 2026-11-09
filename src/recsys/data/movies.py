import pandas as pd


def load_genres(movies: pd.DataFrame) -> dict:
    """{movie_id: set of genres}. '(no genres listed)' becomes an empty set."""
    genres = {}
    for movie_id, genre_text in zip(movies["movie_id"], movies["genres"]):
        if genre_text == "(no genres listed)":
            genres[movie_id] = set()
        else:
            genres[movie_id] = set(genre_text.split("|"))
    return genres