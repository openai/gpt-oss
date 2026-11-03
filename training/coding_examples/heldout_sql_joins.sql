-- 未見之題：以左聯接與聚合核對輪、筒二籍。
CREATE TABLE wheels (name TEXT PRIMARY KEY, power TEXT NOT NULL);
CREATE TABLE buckets (wheel TEXT NOT NULL, amount INTEGER NOT NULL CHECK (amount >= 0));
INSERT INTO wheels VALUES ('東輪', 'river'), ('西輪', 'ox'), ('南輪', 'wind');
INSERT INTO buckets VALUES ('東輪', 4), ('東輪', 2), ('西輪', 3);
SELECT w.name, w.power, COALESCE(SUM(b.amount), 0) AS bucket_total
FROM wheels AS w
LEFT JOIN buckets AS b ON b.wheel = w.name
GROUP BY w.name, w.power
ORDER BY w.name;
