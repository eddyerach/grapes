CREATE TABLE IF NOT EXISTS detections (
    img_id INT PRIMARY KEY,
    img_url VARCHAR(256) UNIQUE NOT NULL,
    last_detected BIGINT NOT NULL,
    last_status SMALLINT,
    grape_count INT,
    detect_url VARCHAR(256)
);


INSERT INTO detections (img_id, img_url, last_detected, last_status, grape_count,detect_url) 
VALUES (%s,%s,%s,%s,%s,%s) 
ON CONFLICT(img_id) DO 
UPDATE SET img_url = EXCLUDED.img_url,
last_detected = EXCLUDED.last_detected,
last_status = EXCLUDED.last_status,
grape_count = EXCLUDED.grape_count,
detect_url = EXCLUDED.detect_url;


SELECT * FROM detections WHERE img_id = %s;