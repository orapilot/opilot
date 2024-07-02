sensors|grep temp1|awk '{sum+=$2} END {print sum/NR}'
