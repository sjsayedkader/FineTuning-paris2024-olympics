CREATE SCHEMA IF NOT EXISTS sk_paris_olympics_2024;

CREATE TABLE IF NOT EXISTS sk_paris_olympics_2024.olympics_ft_training (
  instruction STRING,
  response    STRING,
  source_table STRING
);
