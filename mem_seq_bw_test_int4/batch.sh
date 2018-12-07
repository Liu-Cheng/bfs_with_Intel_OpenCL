# !/bin/sh

arraySize=("4*1024" "8*1024" "16*1024" "32*1024" "64*1024" "128*1024" "256*1024" "512*1024" 
"1024*1024" "2*1024*1024" "4*1024*1024" "8*1024*1024" "16*1024*1024" "32*1024*1024" 
"64*1024*1024" "128*1024*1024" "256*1024*1024")
for i in "${arraySize[@]}"; 
do
	sed -i "/static size_t vector_size /c\static size_t vector_size = $i;" ./src/main.cpp
	make
	run ./bin/mem_bandwidth > tmp.txt
	grep 'Average memory read latency' ./tmp.txt >> latency.txt
done

