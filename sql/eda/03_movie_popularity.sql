-- Distribution of the number of ratings per movie.
WITH per_movie AS (
    SELECT movie_id, COUNT(*) AS n
    FROM ratings
    GROUP BY movie_id
)
SELECT
    MIN(n)                  AS min,
    quantile_cont(n, 0.25)  AS p25,
    median(n)               AS median,
    quantile_cont(n, 0.75)  AS p75,
    quantile_cont(n, 0.99)  AS p99,
    MAX(n)                  AS max,
    ROUND(AVG(n), 1)        AS mean
FROM per_movie;