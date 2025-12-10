CREATE OR REPLACE TABLE workspace.sk_paris_olympics_2024.olympics_ft_training_splits AS
SELECT
  instruction,
  response,
  source_table,
  CASE
    WHEN rand() < 0.9 THEN 'train'
    ELSE 'val'
  END AS split
FROM workspace.sk_paris_olympics_2024.olympics_ft_training;
