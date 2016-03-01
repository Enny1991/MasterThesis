#!/bin/sh

#  validation_timit.sh
#  
#
#  Created by Enea Ceolini on 15/12/15.
#

tsp="1"
model="0 1"
p="0.0 0.2"
enc="150 300"
dec="150 300"
third="0 1"

for t in ${tsp}
do
	for m in ${model}
        do
		for pp in ${p}
        	do
			for n_enc in ${enc}
        		do
				for n_dec in ${dec}
        			do
					for th in ${third}
        				do
						name=$t'_'$m'_'$pp'_'$n_enc'_'$n_dec'_'$th'.txt'
            					python validation_tsp.py ${t} ${m} ${pp} ${n_enc} ${n_dec} ${th} > outputs/$name 
					done
				done
			done
		done
        done
done
