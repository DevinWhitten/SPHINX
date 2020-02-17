#!/bin/sh
#echo "screening TEFF_NET"
#python TEFF_TRAIN.py &
#PID1=$!
#wait $PID1
#echo "TEFF complete"

echo "screening FEH_NET"
python FEH_TRAIN.py &
PID1=$1
wait $PID1
echo "FEH complete"

echo "screening AC_NET"
python AC_TRAIN.py &
PID1=$!
wait $PID1
echo "AC complete"
