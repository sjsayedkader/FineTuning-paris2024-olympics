USE CATALOG workspace;
USE SCHEMA sk_paris_olympics_2024;

WITH athlete_batch AS (
  SELECT
    to_json(
      collect_list(
        named_struct(
          'name',              name,
          'gender',            gender,
          'country',           country,
          'country_long',      country_long,
          'nationality',       nationality,
          'nationality_long',  nationality_long,
          'disciplines',       disciplines,
          'events',            events,
          'height',            height,
          'weight',            weight,
          'birth_date',        birth_date,
          'birth_country',     birth_country,
          'residence_country', residence_country,
          'hobbies',           hobbies,
          'lang',              lang,
          'coach',             coach
        )
      )
    ) AS rows_json
  FROM athletes          -- now unqualified because we did USE SCHEMA
  WHERE name IS NOT NULL
    AND country IS NOT NULL
  LIMIT 20
)
SELECT * FROM athlete_batch;
