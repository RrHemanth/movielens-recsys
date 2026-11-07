-- How many ratings and active users per year?
SELECT
    year(to_timestamp(ts))   AS year,
    COUNT(*)                 AS n_ratings,
    COUNT(DISTINCT user_id)  AS active_users
FROM ratings
GROUP BY year
ORDER BY year;
