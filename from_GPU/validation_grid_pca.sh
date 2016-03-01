#!/bin/sh

#  validation_timit.sh
#  
#
#  Created by Enea Ceolini on 15/12/15.
#

timit="1 2 3 4 5 6"
norm="1"
model="3"
for t in ${timit}
do
    for n in ${norm}
    do
        for m in ${model}
        do
            python validation_grid_pca.py ${t} ${m} ${n}
        done
    done
done
