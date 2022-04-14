#!/bin/bash

ps -ef | grep "kuhn_ppo" | awk '{print $2}' | xargs kill -9
