-- New Year's Day: Temperature and Traffic
SELECT 
  strftime('%Y', date_time) AS year,
  ROUND(AVG((temp - 273.15) * 9.0/5 + 32), 1) AS avg_temp_f,
  ROUND(MIN((temp - 273.15) * 9.0/5 + 32), 1) AS min_temp_f,
  ROUND(MAX((temp - 273.15) * 9.0/5 + 32), 1) AS max_temp_f,
  ROUND(AVG(traffic_volume), 0) AS avg_traffic_volume,
  COUNT(*) AS record_count
FROM Metro_Interstate_Traffic_Volume
WHERE strftime('%m-%d', date_time) = '01-01'  
GROUP BY year
ORDER BY year;

-- Labor Day: temp and traffic
SELECT 
  strftime('%Y', date_time) AS year,
  ROUND(AVG((temp - 273.15) * 9.0/5 + 32), 1) AS avg_temp_f,
  ROUND(MIN((temp - 273.15) * 9.0/5 + 32), 1) AS min_temp_f,
  ROUND(MAX((temp - 273.15) * 9.0/5 + 32), 1) AS max_temp_f,
  ROUND(AVG(traffic_volume), 0) AS avg_traffic_volume,
  COUNT(*) AS record_count
FROM Metro_Interstate_Traffic_Volume
WHERE date(date_time) IN ('2013-09-02','2015-09-07', '2016-09-05', '2017-09-04','2018-09-03')
GROUP BY year
ORDER BY year;



