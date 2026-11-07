-- What share of all ratings goes to the most popular movies?
WITH per_movie AS (
    SELECT movie_id, COUNT(*) AS n
    FROM ratings
    GROUP BY movie_id
),
ranked AS (
    SELECT
        n,
        ROW_NUMBER() OVER (ORDER BY n DESC) AS popularity_rank,
        COUNT(*) OVER ()                    AS n_movies
    FROM per_movie
)
SELECT
    SUM(n) FILTER (WHERE popularity_rank <= n_movies * 0.01) / SUM(n)::DOUBLE AS share_top_1pct_movies,
    SUM(n) FILTER (WHERE popularity_rank <= n_movies * 0.10) / SUM(n)::DOUBLE AS share_top_10pct_movies,
    COUNT(*) FILTER (WHERE n < 10) / COUNT(*)::DOUBLE                         AS frac_movies_under_10_ratings
FROM ranked;