-- How many ratings and users fall in each time period?
-- 1514764800 = 2018-01-01 UTC, 1546300800 = 2019-01-01 UTC
SELECT
    CASE
        WHEN ts < 1514764800 THEN '1_train'
        WHEN ts < 1546300800 THEN '2_val'
        ELSE '3_test'
    END                                        AS split,
    COUNT(*)                                   AS n_ratings,
    COUNT(DISTINCT user_id)                    AS n_users,
    ROUND(AVG((rating >= 4.0)::INT) * 100, 1)  AS pct_relevant
FROM ratings
GROUP BY split
ORDER BY split;