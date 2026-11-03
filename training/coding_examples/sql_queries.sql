-- 以共用表式、窗函式與聚合稽核輪筒之數；只用標準 SQL。
CREATE TABLE readings (
  name TEXT PRIMARY KEY,
  power TEXT NOT NULL,
  turns INTEGER NOT NULL CHECK (turns >= 0),
  buckets INTEGER NOT NULL CHECK (buckets >= 0)
);
INSERT INTO readings (name, power, turns, buckets) VALUES
  ('堰流', 'river', 3, 4),
  ('牛力', 'ox', 2, 2),
  ('風帆', 'wind', 1, 6);
WITH measured AS (
  SELECT name, power, turns * buckets AS lifted,
         SUM(turns * buckets) OVER (PARTITION BY power) AS power_total
  FROM readings
)
SELECT name, lifted, power_total FROM measured ORDER BY name;
