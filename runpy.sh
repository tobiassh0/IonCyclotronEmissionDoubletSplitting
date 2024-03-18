#!/bin/bash

search_dir=/home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_Triton_concentration_high_kperp
declare -i i=0
declare -i t=0
until [ $i -gt 39 ]
do
  i=0
  for entry in "$search_dir"/* #solutions2D_.jld
  do
    #echo "$entry"
    if [ -f "$entry"/w0k0.pkl ]; then
      #echo "File does exist"
      i+=1
    else
      #echo "File does not exist"
      i+=0
    fi
  done
  echo "$i"
  echo "Waited $t min(s)"
  t+=5
  sleep 3
done
echo "($i) pkl files found, running DT-D comparison"
python compare_D_T.py
exit

#declare -i i=0
#declare -i t=5*60
#until [ -f /home/space/phrmsf/Documents/ICE_DS/JET26148/default_params_with_no_Tritons_high_kperp/run_2.07_0.0_-0.646_0.01_0.01_25.0_3.5_0.975_1.0_4.0_2.524544123190745e19_0.00015_2048/w0k0.pkl ]
#do
#  echo "Waiting" $i "mins"
#  sleep "$t"
#  i+=5
#done
#echo "pkl files found"
#python compare_D_T.py
#exit
