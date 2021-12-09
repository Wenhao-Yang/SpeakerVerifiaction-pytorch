#!/usr/bin/env bash


while true;do
    sleep 900
    ppp=`ifconfig | grep ppp0 | wc -l`

    if [ $ppp -eq 1 ]; then
        /etc/init.d/networking restart
    fi
done