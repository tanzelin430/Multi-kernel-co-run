#!/bin/bash

output_file="outputnewnewnewnew.csv"

echo "torchbatch:" > $output_file

# for i in {1,5,10,15,20,25,30,35,40}; do
#     echo "Running with num_requests = $i"
#     python mlp_torch_batch.py $i >> $output_file
# done
echo "torchsplit:" >> $output_file
for i in {1,5,10,15,20}; do
    echo "Running with num_requests = $i"
    python mlp_torch_split.py $i >> $output_file
    sleep 5
done
# echo "opsbatch:" >> $output_file
# for i in {1,5,10,15,20,25,30,35,40}; do
#     echo "Running with num_requests = $i"
#     python mlp_ops_batch.py $i >> $output_file
# done
# echo "opssplit:" >> $output_file
# for i in {1,5,10,15,20,25,30,35,40}; do
#     echo "Running with num_requests = $i"
#     python mlp_ops_split.py $i >> $output_file
# done
