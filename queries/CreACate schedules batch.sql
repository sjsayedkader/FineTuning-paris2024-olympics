WITH sample AS (
  SELECT
    start_date,
    end_date,
    day,
    discipline,
    event,
    event_medal,
    phase,
    gender,
    event_type,
    venue,
    venue_code
  FROM schedules
  WHERE event IS NOT NULL
  ORDER BY rand()
  LIMIT 8
),
schedules_batch AS (
  SELECT
    to_json(
      collect_list(
        named_struct(
          'start_date', start_date,
          'end_date',   end_date,
          'day',        day,
          'discipline', discipline,
          'event',      event,
          'event_medal',event_medal,
          'phase',      phase,
          'gender',     gender,
          'event_type', event_type,
          'venue',      venue,
          'venue_code', venue_code
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
about the schedule of Paris 2024 Olympic events.

The data is a JSON array of objects with fields:
- start_date, end_date, day
- discipline, event, event_medal, phase, gender, event_type
- venue, venue_code

Rows:
' || rows_json || '

Task:
- Generate 6 question-answer pairs, such as:
  - "On which day is <event> scheduled?"
  - "When does <event> start and end?"
  - "At which venue is <event> held?"
  - "What is the phase or event type of <event>?"
- Answers must be short factual sentences using ONLY the values from the rows.
- Output STRICTLY in JSON Lines format:
  {\"input\": \"<question>\", \"output\": \"<answer>\"}
' AS full_prompt
  FROM schedules_batch
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
SELECT instruction, response, 'schedules' AS source_table
FROM parsed;
