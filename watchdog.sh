#!/bin/bash
# Backs up the v2 checkpoint every 15 min; when job 5522634 leaves the queue,
# rebuilds the evaluation report from whatever rows exist.
cd "/mnt/aiongpfs/users/pkonda/pubmedqa_baseline_rag/baseline rag"
PY=/home/users/pkonda/miniconda3/envs/rag_thesis/bin/python
JID=5522634
LOG=watchdog.log
echo "[$(date +%T)] watchdog started for job $JID" >> $LOG
while squeue -j $JID -h -o "%T" 2>/dev/null | grep -q .; do
  N=$(wc -l < outputs_v2/advanced_answers.jsonl 2>/dev/null || echo 0)
  ./backup_checkpoint_v2.sh >> $LOG 2>&1
  echo "[$(date +%T)] running, $N/1000 rows" >> $LOG
  sleep 900
done
N=$(wc -l < outputs_v2/advanced_answers.jsonl 2>/dev/null || echo 0)
echo "[$(date +%T)] JOB ENDED with $N rows. sacct:" >> $LOG
sacct -j $JID --format=State%14,ExitCode,Elapsed -n >> $LOG 2>&1
./backup_checkpoint_v2.sh >> $LOG 2>&1
echo "[$(date +%T)] building partial report..." >> $LOG
cd src && $PY build_report.py --answers ../outputs_v2/advanced_answers.jsonl    --out ../outputs_v2/full_evaluation_report_partial.json >> ../$LOG 2>&1
echo "[$(date +%T)] DONE. report at outputs_v2/full_evaluation_report_partial.json" >> ../$LOG
