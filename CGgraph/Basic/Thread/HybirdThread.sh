#! /bin/bash

HYPERTHREADING=1

function toggleHyperThreading() {
 for CPU in /sys/devices/system/cpu/cpu[0-9]*; do
   CPUID=`basename $CPU | cut -b4-`
   echo -en "CPU: $CPUID\t"
   [ -e $CPU/online ] && echo "1" > $CPU/online
   THREAD1=`cat $CPU/topology/thread_siblings_list | cut -f1 -d,`
   if [ $CPUID = $THREAD1 ]; then
     echo "-> enable"
     [ -e $CPU/online ] && echo "1" > $CPU/online
    else
    if [ "$HYPERTHREADING" -eq "0" ]; then echo "-> disabled"; else echo "-> enabled"; fi
     echo "$HYPERTHREADING" > $CPU/online
    fi
 done
}

function enabled() {
    echo "---------------------------"
	echo "Ready to open Hybird Thread"
	echo "---------------------------"
    HYPERTHREADING=1
    toggleHyperThreading
	echo -en "[Complete]:Enabling HyperThreading\n"
}

function disabled() {
    echo "----------------------------"
	echo "Ready to close Hybird Thread"
	echo "----------------------------"
    HYPERTHREADING=0
    toggleHyperThreading
	echo -en "[Complete]:Disabling HyperThreading\n"
}


while true; do
    #read -p "Type in e to enable or d disable hyperThreading or q to quit [e/d/q] ?" ed
    case ${1} in
        [Ee]* ) enabled; break;;
        [Dd]* ) disabled;exit;;
        [Qq]* ) exit;;
        * ) echo "Please answer e for enable or d for disable hyperThreading."; break;;
    esac
done
