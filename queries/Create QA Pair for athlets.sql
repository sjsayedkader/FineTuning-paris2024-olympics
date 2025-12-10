USE CATALOG workspace;
USE SCHEMA sk_paris_olympics_2024;

WITH sample AS (
  SELECT
    name,
    gender,
    country,
    country_long,
    nationality,
    nationality_long,
    disciplines,
    events,
    height,
    weight,
    birth_date,
    birth_country,
    residence_country
  FROM athletes
  WHERE name IS NOT NULL
    AND country IS NOT NULL
  ORDER BY rand()          -- 👈 randomize each run
  LIMIT 8
),
athlete_batch AS (
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
          'residence_country', residence_country
        )
      )
    ) AS rows_json
  FROM sample
),
prompt AS (
  SELECT
    """
You are a data transformation assistant.
Use ONLY the data provided below to generate Q&A pairs
about athletes at the Paris 2024 Olympic Games.

Rows:
""" || rows_json || """

Task:
- Generate 6 diverse question-answer pairs.
- Questions should be natural (who do they represent, what discipline, nationality, height, etc.).
- Answers must be short factual sentences using ONLY the fields from the rows.
- If a fact is missing for a particular athlete, do NOT ask about it.
- Output STRICTLY in JSON Lines format:
  {\"input\": \"<question>\", \"output\": \"<answer>\"}
- Do not add any explanation before or after the JSON Lines.
""" AS full_prompt
  FROM athlete_batch
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
SELECT instruction, response, 'athletes' AS source_table
FROM parsed;
