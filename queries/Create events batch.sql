USE CATALOG workspace;
USE SCHEMA sk_paris_olympics_2024;

WITH sample AS (
  SELECT
    event,
    tag,
    sport,
    sport_code
  FROM events
  WHERE event IS NOT NULL
  ORDER BY rand()
  LIMIT 8
),
events_batch AS (
  SELECT
    to_json(
      collect_list(
        named_struct(
          'event',      event,
          'tag',        tag,
          'sport',      sport,
          'sport_code', sport_code
        )
      )
    ) AS rows_json
  FROM sample
),
prompt AS (
  SELECT
'You are a data transformation assistant.
Use ONLY the data provided below to generate Q&A pairs
about events at the Paris 2024 Olympics.

The data is a JSON array of objects with fields:
- event, tag, sport, sport_code

Rows:
' || rows_json || '

Task:
- Generate 6 question-answer pairs, such as:
  - "In which sport is the event <event>?"
  - "What is the tag or category for <event>?"
  - "What is the sport code for <sport>?"
- Answers must be short factual sentences using ONLY the values from the rows.
- Output STRICTLY in JSON Lines format:
  {\"input\": \"<question>\", \"output\": \"<answer>\"}
' AS full_prompt
  FROM events_batch
),
raw_response AS (
  SELECT ai_query('databricks-meta-llama-3-1-8b-instruct', full_prompt) AS jsonl_text
  FROM prompt
),
split_lines AS (
  SELECT trim(line) AS line
  FROM raw_response
  LATERAL VIEW explode(split(jsonl_text, '\n')) AS line
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
SELECT instruction, response, 'events' AS source_table
FROM parsed;
