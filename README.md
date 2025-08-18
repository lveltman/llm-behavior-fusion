# llm-behavior-fusion

screen -S train -L -Logfile train.log accelerate launch \
  --config_file config/ds_config.yaml train.py