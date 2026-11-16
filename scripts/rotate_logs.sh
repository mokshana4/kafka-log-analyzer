#!/bin/bash
LOG="./logs/app_$(date +%F).log"
if [ -f "$LOG" ]; then
    gzip "$LOG"
    echo "Rotated → $LOG.gz"
else
    echo "No log file found to rotate."
fi