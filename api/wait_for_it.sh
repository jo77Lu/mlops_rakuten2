#!/usr/bin/env bash
# wait-for-it.sh: Waits until a given host and port are ready.

host=$1
cmd="$@"
timeout=600
start_time=$(date+%s)

until $(curl --output /dev/null --silent --head --fail http://$host); do
  printf '.'

  current_time=$(date+%s)
  elapsed_time=$((current_time - start_time))

  if [ $elapsed_time -ge $timeout ]; then
    echo "Timeout reached after $timeout seconds. Exiting."
    exit 1
  sleep 5
done

exec $cmd