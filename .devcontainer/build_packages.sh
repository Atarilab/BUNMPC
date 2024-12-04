#!/bin/bash

ORANGE=$'\e[0;33m'
GREEN=$'\e[0;32m'
RED=$'\e[0;31m'
NC=$'\e[0m'

# do one update
sudo apt update

###### BICONVEX_MPC DEPENDENCIES ######

# Check if bullet_utils is downloaded
if [ -d "/home/atari_ws/bullet_utils" ]; then
    echo "bullet_utils found"
    echo "${ORANGE}installing bullet utils...${NC}"
    cd /home/atari_ws/bullet_utils
    sudo pip3 install .
    echo "${GREEN}Succesfully installed bullet_utils${NC}"

else  # if bullet_utils is not installed... 
    echo "${RED}bullet_utils NOT found${NC}"
    echo "please download bullet_utils from https://github.com/machines-in-motion/bullet_utils"
fi

# check if robot_properties_solo is downloaded
if [ -d "/home/atari_ws/robot_properties_solo" ]; then
    echo "robot_properties_solo found"
    echo "${ORANGE}installing robot_properties_solo...${NC}"
    cd /home/atari_ws/robot_properties_solo
    sudo pip3 install . 
    echo "${GREEN}Succesfully installed robot_properties_solo${NC}"

else
    echo "${RED}robot_properties_solo NOT found${NC}"
    echo "please download robot_properties_solo from https://github.com/open-dynamic-robot-initiative/robot_properties_solo"
fi

# check if robot_properties_go2 is downloaded
if [ -d "/home/atari_ws/robot_properties_go2" ]; then
    echo "robot_properties_solo found"
    echo "${ORANGE}installing robot_properties_go2 ...${NC}"
    cd /home/atari_ws/robot_properties_go2
    sudo pip3 install . 
    echo "${GREEN}Succesfully installed robot_properties_go2${NC}"

else
    echo "${RED}robot_properties_go2 NOT found${NC}"
fi


###### Iterative Supervised Learning ######

# Check if ISL directory exists
if [ -d "/home/atari_ws/iterative_supervised_learning" ]; then
    echo "iterative_supervised_learning found"

    # check if user pulled the repo correctly with non-empty extern directory
    if [ -z "$(ls -A /home/atari_ws/iterative_supervised_learning/extern)" ]; then
        echo "${RED}extern dir is empty! Please pull the git repo correctly!{NC}"
        # echo "command: git clone --recurse-submodules https://github.com/majidkhadiv/goal_conditioned_behavioral_cloning.git"
        exit 10

    else
        echo "${ORANGE}Building iterative_supervised_learning...${NC}"
        
        # check if iterative_supervised_learning is built before
        if [ -d "/home/atari_ws/iterative_supervised_learning/build" ]; then

            echo "build dir in iterative_supervised_learning found"
            echo "deleting build file..."

            # Delete any previous build files in iterative_supervised_learning
            if [ -d "/home/atari_ws/iterative_supervised_learning/build" ]; then
                cd /home/atari_ws/iterative_supervised_learning
                sudo rm -rf build
            fi
        fi
        
        source /home/atari/.bashrc
        cd /home/atari_ws/iterative_supervised_learning
        mkdir build && cd build

        cmake .. -DCMAKE_BUILD_TYPE=Release 
        make install -j8
            
        echo "${GREEN}Succesfully build iterative_supervised_learning${NC}"
        
    fi

else
    echo "${RED}iterative_supervised_learning NOT found${NC}"
    echo "please download iterative_supervised_learning from https://github.com/majidkhadiv/goal_conditioned_behavioral_cloning.git"
    echo "command: git clone --recurse-submodules https://github.com/majidkhadiv/goal_conditioned_behavioral_cloning.git"
fi


###### BICONVEX_MPC ######

# # Check if biconvex_mpc directory exists
# if [ -d "/home/atari_ws/biconvex_mpc" ]; then
#     echo "biconvex_mpc found"

#     # check if user pulled the repo correctly with non-empty extern directory
#     if [ -z "$(ls -A /home/atari_ws/biconvex_mpc/extern)" ]; then
#         echo "${RED}extern dir is empty! Please pull the git repo correctly!{NC}"
#         echo "command: git clone --recurse-submodules https://github.com/machines-in-motion/biconvex_mpc.git"
#     else
#         # check if biconvex_mpc is built before
#         if [ -d "/home/atari_ws/biconvex_mpc/build" ]; then
#             echo "build dir in biconvex_mpc found"

#             read -p "${ORANGE}delete and rebuild biconvex_mpc? This will take a few minutes${NC} (y/n)?" CONT
#             if [ "$CONT" = "y" ]; then
#                 echo "deleting build file..."

#                 # Delete any previous build files in biconvex_mpc
#                 if [ -d "/home/atari_ws/biconvex_mpc/build" ]; then
#                     cd /home/atari_ws/biconvex_mpc
#                     sudo rm -rf build
#                 fi

#                 # Build biconvex_mpc
#                 echo "Building biconvex_mpc"

#                 source /home/atari/.bashrc
#                 cd /home/atari_ws/biconvex_mpc
#                 mkdir build && cd build

#                 cmake .. -DCMAKE_BUILD_TYPE=Release 
#                 make install -j16
                    
#                 echo "${GREEN}Succesfully build biconvex_mpc${NC}"
#             else
#                 echo "biconvex_mpc will not be rebuilt. "
#             fi

#         else
#             # Build biconvex_mpc
#             echo "Building biconvex_mpc"

#             source /home/atari/.bashrc
#             cd /home/atari_ws/biconvex_mpc
#             mkdir build && cd build

#             cmake .. -DCMAKE_BUILD_TYPE=Release 
#             make install -j16
                
#             echo "${GREEN}Succesfully build biconvex_mpc${NC}"
#         fi
#     fi

# else
#     echo "${RED}biconvex_mpc NOT found${NC}"
#     echo "please download biconvex_mpc from https://github.com/machines-in-motion/biconvex_mpc"
#     echo "command: git clone --recurse-submodules https://github.com/machines-in-motion/biconvex_mpc.git"
# fi

###### Goal Conditioned Behavioral Cloning ######

# # Check if GCBC directory exists
# if [ -d "/home/atari_ws/goal_conditioned_behavioral_cloning" ]; then
#     echo "goal_conditioned_behavioral_cloning found"

#     # check if user pulled the repo correctly with non-empty extern directory
#     if [ -z "$(ls -A /home/atari_ws/goal_conditioned_behavioral_cloning/extern)" ]; then
#         echo "${RED}extern dir is empty! Please pull the git repo correctly!{NC}"
#         echo "command: git clone --recurse-submodules https://github.com/majidkhadiv/goal_conditioned_behavioral_cloning.git"
#         exit 10
#     else
#         # check if  goal_conditioned_behavioral_cloning is built before
#         if [ -d "/home/atari_ws/goal_conditioned_behavioral_cloning/build" ]; then
#             echo "build dir in goal_conditioned_behavioral_cloning found"

#             read -p "${ORANGE}delete and rebuild goal_conditioned_behavioral_cloning? This will take a few minutes${NC} (y/n)?" CONT
#             if [ "$CONT" = "y" ]; then
#                 echo "deleting build file..."

#                 # Delete any previous build files in goal_conditioned_behavioral_cloning
#                 if [ -d "/home/atari_ws/goal_conditioned_behavioral_cloning/build" ]; then
#                     cd /home/atari_ws/goal_conditioned_behavioral_cloning
#                     sudo rm -rf build
#                 fi

#                 # Build goal_conditioned_behavioral_cloning
#                 echo "Building goal_conditioned_behavioral_cloning"

#                 source /home/atari/.bashrc
#                 cd /home/atari_ws/goal_conditioned_behavioral_cloning
#                 mkdir build && cd build

#                 cmake .. -DCMAKE_BUILD_TYPE=Release 
#                 make install -j16
                    
#                 echo "${GREEN}Succesfully build goal_conditioned_behavioral_cloning${NC}"
#             else
#                 echo "goal_conditioned_behavioral_cloning will not be rebuilt. "
#             fi

#         else
#             # Build goal_conditioned_behavioral_cloning
#             echo "Building goal_conditioned_behavioral_cloning"

#             source /home/atari/.bashrc
#             cd /home/atari_ws/goal_conditioned_behavioral_cloning
#             mkdir build && cd build

#             cmake .. -DCMAKE_BUILD_TYPE=Release 
#             make install -j16
                
#             echo "${GREEN}Succesfully build goal_conditioned_behavioral_cloning${NC}"
#         fi
#     fi

# else
#     echo "${RED}goal_conditioned_behavioral_cloning NOT found${NC}"
#     echo "please download goal_conditioned_behavioral_cloning from https://github.com/majidkhadiv/goal_conditioned_behavioral_cloning.git"
#     echo "command: git clone --recurse-submodules https://github.com/majidkhadiv/goal_conditioned_behavioral_cloning.git"
# fi




