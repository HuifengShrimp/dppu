#!/bin/bash

CMD_PREFIX="[dppu]"
CMD="export PATH=\$PATH:`pwd`/bin"

# install dppu into system path (optional)
if ! command -v dppu & > /dev/null
then
    echo "dppu not found, adding dppu to ~/.bashrc"
    echo $CMD >> ~/.bashrc
else
    echo "dppu found in: `which dppu` "
fi

# build ppu docker image from 
cd ppu/docker &&
docker build -t ppu-dev -f ppu-gcc11-anolis-dev.Dockerfile .

# echo "DPPU_PATH=$(pwd)" > default.conf
# if [ -z "$PPU_PATH" ]
# then
#     echo -e "\033[33m[warning]\033[0m" "ppu path not found, initializing ..."
# fi
