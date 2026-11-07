-- How big is the data, how sparse is it, and what time span does it cover?
SELECT
    COUNT(*)                  AS n_ratings,
    COUNT(DISTINCT user_id)   AS n_users,
    COUNT(DISTINCT movie_id)  AS n_movies_rated,
    COUNT(*)::DOUBLE
        / (COUNT(DISTINCT user_id)::DOUBLE * COUNT(DISTINCT movie_id)) AS density,
    to_timestamp(MIN(ts))     AS first_rating,
    to_timestamp(MAX(ts))     AS last_rating
FROM ratings;