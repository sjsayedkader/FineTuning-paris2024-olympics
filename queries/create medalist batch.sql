USE CATALOG workspace;
USE SCHEMA sk_paris_olympics_2024;

WITH sample AS (
  SELECT
    medal_date,
    medal_type,
    name,
    gender,
    country_code,
    country,
    country_long,
    team,
    team_gender,
    discipline,
    event,
    event_type
  FROM medallists
  WHERE is_medallist = true
    AND name IS NOT NULL
    AND event IS NOT NULL
  ORDER BY rand()
  LIMIT 8
),
medallists_batch AS (
  SELECT
    to_json(
      collect_list(
        named_struct(
          'medal_date',   medal_date,
          'medal_type',   medal_type,
          'name',         name,
          'gender',       gender,
          'country_code', country_code,
          'country',      country,
          'country_long', country_long,
          'team',         team,
          'team_gender',  team_gender,
          'discipline',   discipline,
          'event',        event,
          'event_type',   event_type
        )
      )
    ) AS rows_json
  FROM sample
),
prompt AS (
  SELECT
    'You are a data transformation assistant.
Use ONLY the data provided below to generate Q&A pairs
about Paris 2024 Olympic medallists.

The data is a JSON array of objects with fields:
- medal_date, medal_type, name, gender
- country_code, country, country_long
- team, team_gender, discipline, event, event_type

Rows:
' || rows_json || '

Task:
- Generate 6 question-answer pairs.
- Mix question types, for example:
  - "Who won the <medal_type> medal in <event>?"
  - "Which country does <name> represent?"
  - "For which team did <name> compete?"
  - "In which discipline or event type did <name> win a medal?"
- Answers must be short factual sentences using ONLY the values from the rows.
- Output STRICTLY in JSON Lines format:
  {\"input\": \"<question>\", \"output\": \"<answer>\"}
- No extra explanation.'
    AS full_prompt
  FROM medallists_batch
),
raw_response AS (
  SELECT ai_query('databricks-meta-llama-3-1-8b-instruct', full_prompt) AS jsonl_text
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
SELECT instruction, response, 'medallists' AS source_table
FROM parsed;