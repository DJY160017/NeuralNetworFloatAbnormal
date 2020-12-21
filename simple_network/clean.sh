#!/bin/bash
RED_COLOR='\E[1;31m'  #红
GREEN_COLOR='\E[1;32m' #绿
YELOW_COLOR='\E[1;33m' #黄
RES='\E[0m'

cd /home/byron/Documents/workspace/iRRAM_improved_byron
echo -e "${YELOW_COLOR}clean before out{RES}"
rm -rf /usr/local/include/iRRAM*
rm -rf /usr/local/lib/libiRRAM.*
echo -e "${GREEN_COLOR}clean end and run install{RES}"

./QUICKINSTALL_run_me

