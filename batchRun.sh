#!/bin/bash

if [ "${10}" = "1" ]; then
    mkdir -p results_kan
    filename="results_kan/${1}.txt"
else
    mkdir -p results
    filename="results/${1}.txt"
fi

rm -f "$filename"
touch "$filename"
echo $filename
for arg in "$@"; do
  echo "Arg $n: $arg"
  ((n++))
done
echo ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9} ${10}
i=1
#for i in {1..10}
#do
python3 -W ignore main.py $i ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9} ${10} | tee >(egrep "para|systemSelector|fit|NRMSE|f1|elapsed|evaluating|encoder|decoder|bridge" >> "$filename")
#done
