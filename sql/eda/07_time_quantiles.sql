-- At what dates do 80%, 90% and 95% of all ratings fall before?
SELECT
    to_timestamp(quantile_disc(ts, 0.80)) AS date_80pct,
    to_timestamp(quantile_disc(ts, 0.90)) AS date_90pct,
    to_timestamp(quantile_disc(ts, 0.95)) AS date_95pct
FROM ratings;