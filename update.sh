#!/bin/bash
# Quarterly S&P 500 data update — called by cron
cd "/Users/romingandhi/Desktop/Long:Short S&P 500 Strategy"
/usr/bin/python3 data_preprocessing/data_pipeline.py --update >> logs/cron.log 2>&1
