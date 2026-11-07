-- How often is each rating value used?
SELECT
    rating,
    COUNT(*) AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM ratings
GROUP BY rating
ORDER BY rating;