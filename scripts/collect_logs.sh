#!/bin/bash
SOURCE="./logs/sample_logs.txt"
DEST="./logs/app_$(date +%F).log"
cp "$SOURCE" "$DEST"
echo "Logs collected → $DEST"