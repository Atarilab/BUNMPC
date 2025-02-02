#!/bin/bash

ORANGE=$'\e[0;33m'
GREEN=$'\e[0;32m'
RED=$'\e[0;31m'
NC=$'\e[0m'


###### BICONVEX_MPC ######

# Check if biconvex_mpc directory exists
if [ -d "/home/atari_ws/biconvex_mpc" ]; then
    echo "biconvex_mpc found"

    # check if user pulled the repo correctly with non-empty extern directory
    if [ -z "$(ls -A /home/atari_ws/biconvex_mpc/extern)" ]; then
        echo "${RED}extern dir is empty! Please pull the git repo correctly!{NC}"
        echo "command: git clone --recurse-submodules https://github.com/machines-in-motion/biconvex_mpc.git"
    else
        # check if biconvex_mpc is built before
        if [ -d "/home/atari_ws/biconvex_mpc/build" ]; then
            echo "build dir in biconvex_mpc found"

            read -p "${ORANGE}delete and rebuild biconvex_mpc? This will take a few minutes${NC} (y/n)?" CONT
            if [ "$CONT" = "y" ]; then
                echo "deleting build file..."

                # Delete any previous build files in biconvex_mpc
                if [ -d "/home/atari_ws/biconvex_mpc/build" ]; then
                    cd /home/atari_ws/biconvex_mpc
                    sudo rm -rf build
                fi

                # Build biconvex_mpc
                echo "Building biconvex_mpc"

                source /home/atari/.bashrc
                cd /home/atari_ws/biconvex_mpc
                mkdir build && cd build

                cmake .. -DCMAKE_BUILD_TYPE=Release 
                make install -j16
                    
                echo "${GREEN}Succesfully build biconvex_mpc${NC}"
            else
                echo "biconvex_mpc will not be rebuilt. "
            fi

        else
            # Build biconvex_mpc
            echo "Building biconvex_mpc"

            source /home/atari/.bashrc
            cd /home/atari_ws/biconvex_mpc
            mkdir build && cd build

            cmake .. -DCMAKE_BUILD_TYPE=Release 
            make install -j16
                
            echo "${GREEN}Succesfully build biconvex_mpc${NC}"
        fi
    fi

else
    echo "${RED}biconvex_mpc NOT found${NC}"
    echo "please download biconvex_mpc from https://github.com/machines-in-motion/biconvex_mpc"
    echo "command: git clone --recurse-submodules https://github.com/machines-in-motion/biconvex_mpc.git"
fi



