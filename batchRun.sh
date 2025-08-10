#!/bin/bash

if [ "$9" = "1" ]; then
    filename="results/${1}_KAN.txt"
else
    filename="results/${1}.txt"
fi

rm -f "$filename"
touch "$filename"

echo $2 $3 $4 $5 $6 $7 $8 $9
for i in {1..10}
do
	python3 -W ignore main.py $i $2 $3 $4 $5 $6 $7 $8 $9 | tee >(egrep "para|systemSelector|fit|NRMSE|f1|elapsed|evaluating|encoder|decoder|bridge" >> "$filename")
done
