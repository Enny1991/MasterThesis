#!/bin/sh

#  validation_timit.sh
#  
#
#  Created by Enea Ceolini on 15/12/15.
#

timit="1 2 3 4 5 6 7 8 9 10"
norm="1 0"
model="0 1 2"
for t in ${timit}
do
    for n in ${norm}
    do
        for m in ${model}
        do
            python 100_validation_timit.py ${t} ${m} ${n}
        done
    done
done
