#!/bin/sh

#  validation_timit.sh
#  
#
#  Created by Enea Ceolini on 15/12/15.
#

grid="1 2 3 4 5 6"
norm="0"
model="1"

for t in ${grid}
do
    for n in ${norm}
    do
        for m in ${model}
        do
            python validation_grid.py ${t} ${m} ${n}
        done
    done
done
