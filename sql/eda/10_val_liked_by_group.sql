-- In 2018, how many movies does a typical new vs returning user like (rate 4+)?
WITH train_users AS (
    SELECT DISTINCT user_id FROM read_parquet('data/processed/train.parquet')
),
val AS (
    SELECT * FROM read_parquet('data/processed/val.parquet')
)
SELECT
    CASE WHEN t.user_id IS NULL THEN 'new' ELSE 'returning' END   AS user_group,
    COUNT(DISTINCT v.user_id)                                     AS n_users,
    COUNT(*)                                                      AS n_ratings,
    ROUND(COUNT(*) FILTER (WHERE v.rating >= 4.0)::DOUBLE
          / COUNT(DISTINCT v.user_id), 1)                         AS avg_liked_per_user
FROM val v
LEFT JOIN train_users t ON v.user_id = t.user_id
GROUP BY user_group;