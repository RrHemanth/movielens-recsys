-- How many val/test users are known (seen earlier) vs new (cold start)?
-- 1514764800 = 2018-01-01 UTC, 1546300800 = 2019-01-01 UTC
WITH per_user AS (
    SELECT
        user_id,
        bool_or(ts <  1514764800)                         AS in_train,
        bool_or(ts >= 1514764800 AND ts < 1546300800)     AS in_val,
        bool_or(ts >= 1546300800)                         AS in_test
    FROM ratings
    GROUP BY user_id
)
SELECT
    COUNT(*) FILTER (WHERE in_val)                                AS val_users,
    COUNT(*) FILTER (WHERE in_val AND in_train)                   AS val_known,
    COUNT(*) FILTER (WHERE in_val AND NOT in_train)               AS val_new,
    COUNT(*) FILTER (WHERE in_test)                               AS test_users,
    COUNT(*) FILTER (WHERE in_test AND (in_train OR in_val))      AS test_known,
    COUNT(*) FILTER (WHERE in_test AND NOT in_train AND NOT in_val) AS test_new
FROM per_user;