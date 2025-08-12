#!/bin/bash
for day in {1..19}
do
  echo "Update for July $day" >> file.txt
  git add file.txt
  GIT_COMMITTER_DATE="2024-07-$(printf "%02d" $day)T10:00:00" \
  git commit --date="2024-07-$(printf "%02d" $day)T10:00:00" \
  -m "Update for July $day"
done

git push origin main


