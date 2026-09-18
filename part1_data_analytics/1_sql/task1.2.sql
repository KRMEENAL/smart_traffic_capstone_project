select strftime('%Y', date_time) AS year, sum(traffic_volume) from Metro_Interstate_Traffic_Volume group by year

WITH yearly AS (
  SELECT 
    strftime('%Y', date_time) AS year,
    SUM(traffic_volume) AS total_volume
  FROM Metro_Interstate_Traffic_Volume
  GROUP BY year
)
SELECT 
  year,
  total_volume,
  total_volume - LAG(total_volume) OVER (ORDER BY year) AS change,
  CASE 
    WHEN total_volume > LAG(total_volume) OVER (ORDER BY year) THEN 'Increase'
    WHEN total_volume < LAG(total_volume) OVER (ORDER BY year) THEN 'Decrease'
    ELSE 'No prior year'
  END AS trend
FROM yearly
ORDER BY year;