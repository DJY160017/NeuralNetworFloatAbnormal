#!/bin/bash
RED_COLOR='\E[1;31m'  #红
GREEN_COLOR='\E[1;32m' #绿
YELOW_COLOR='\E[1;33m' #黄
RES='\E[0m'

target=$(grep -o "project(.*)" CMakeLists.txt | cut -d\( -f2 | cut -d\) -f1)
echo -e "${GREEN_COLOR}find target is ${target}${RES}"

cd build/
echo -e  "${GREEN_COLOR}clean build and set ulimit${RES}"
rm -rf ../log.txt
rm -rf *
ulimit -c unlimited
echo -e  "${GREEN_COLOR}enter dir build${RES}"
cmake ..
echo -e "${GREEN_COLOR}cmake finish${RES}"
make
echo -e "${GREEN_COLOR}make finish${RES}"
./${target} > ../log.txt
