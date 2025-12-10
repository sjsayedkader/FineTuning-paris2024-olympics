USE CATALOG workspace;
USE SCHEMA sk_paris_olympics_2024;

WITH sample AS (
  SELECT
    country_code,
    country,
    country_long,
    `Gold Medal`   AS gold,
    `Silver Medal` AS silver,
    `Bronze Medal` AS bronze,
    Total          AS total
  FROM medals_total
  WHERE country IS NOT NULL
  ORDER BY rand()
  LIMIT 8
),
medal_batch AS (
  SELECT
    to_json(
      collect_list(
        named_struct(
          'country_code', country_code,
          'country',      country,
          'country_long', country_long,
          'gold',         gold,
          'silver',       silver,
          'bronze',       bronze,
          'total',        total
        )
      )
    ) AS rows_json
  FROM sample
),
prompt AS (
  SELECT
    'You are a data transformation assistant.
Use ONLY the data provided below to generate Q&A pairs
about Paris 2024 Olympic medal standings by country.

The data is a JSON array of country objects with fields:
- country_code, country, country_long
- gold, silver, bronze, total

Rows:
' || rows_json || '

Task:
- Generate 6 question-answer pairs.
- Mix question types, for example:
  - "How many medals did <country> win at the Paris 2024 Olympics?"
  - "How many gold medals did <country> win?"
  - "How many silver/bronze medals did <country> win?"
  - "What is the NOC code for <country>?"
- Answers must be short factual sentences using ONLY the values from the rows.
- Output STRICTLY in JSON Lines format:
  {\"input\": \"<question>\", \"output\": \"<answer>\"}
- Do not add any explanation before or after the JSON Lines.'
    AS full_prompt
  FROM medal_batch
),
raw_response AS (
  SELECT
    ai_query(
      'databricks-meta-llama-3-1-8b-instruct',
      full_prompt
    ) AS jsonl_text
  FROM prompt
),
split_lines AS (
  SELECT trim(line) AS line
  FROM raw_response
  LATERAL VIEW explode(
    split(
      jsonl_text,
      '\n'
    )
  ) AS line
  WHERE trim(line) != ''
),
parsed AS (
  SELECT
    get_json_object(line, '$.input')  AS instruction,
    get_json_object(line, '$.output') AS response
  FROM split_lines
  WHERE line LIKE '{%'
    AND get_json_object(line, '$.input')  IS NOT NULL
    AND get_json_object(line, '$.output') IS NOT NULL
)

INSERT INTO olympics_ft_training
SELECT instruction, response, 'medals_total' AS source_table
FROM parsed;