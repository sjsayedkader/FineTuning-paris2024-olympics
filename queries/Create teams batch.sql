WITH sample AS (
  SELECT
    code,
    team,
    team_gender,
    country_code,
    country,
    country_long,
    discipline,
    events,
    num_athletes,
    num_coaches
  FROM teams
  WHERE team IS NOT NULL
  ORDER BY rand()
  LIMIT 8
),
teams_batch AS (
  SELECT
    to_json(
      collect_list(
        named_struct(
          'code',         code,
          'team',         team,
          'team_gender',  team_gender,
          'country_code', country_code,
          'country',      country,
          'country_long', country_long,
          'discipline',   discipline,
          'events',       events,
          'num_athletes', num_athletes,
          'num_coaches',  num_coaches
        )
      )
    ) AS rows_json
  FROM sample
),
prompt AS (
  SELECT
    '
You are a data transformation assistant.
Use ONLY the data provided below to generate Q&A pairs
about national teams at the Paris 2024 Olympic Games.

The data is a JSON array of objects with fields:
- code, team, team_gender
- country_code, country, country_long
- discipline, events
- num_athletes, num_coaches

Rows:
' || rows_json || '

Task:
- Generate 6 question-answer pairs, such as:
  - "Which country does the team <team> represent?"
  - "Is the team <team> mens, womens, or mixed?"
  - "In which discipline or events does <team> compete?"
  - "How many athletes or coaches are on <team>?"
  - "What is the team code for <team>?"
- Answers must be short factual sentences using ONLY the values from the rows.
- Output STRICTLY in JSON Lines format:
  {\"input\": \"<question>\", \"output\": \"<answer>\"}
' AS full_prompt
  FROM teams_batch
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
SELECT instruction, response, 'teams' AS source_table
FROM parsed;
