#! /bin/bash
set -e

# Settings from environment
UDACITY_SOURCE=${UDACITY_SOURCE:-`pwd`}
CONTAINER_NAME="capstone"
UDACITY_IMAGE=${UDACITY_IMAGE:-tantony/carnd-capstone-cuda}
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

if [ "$(docker ps | grep ${CONTAINER_NAME})" ]; then
  echo "Attaching to running container..."
  nvidia-docker exec -e DISPLAY=$DISPLAY -it ${CONTAINER_NAME} bash $@
else
  nvidia-docker run -e DISPLAY=$DISPLAY -e XAUTHORITY=${XAUTH} --name ${CONTAINER_NAME} --rm -it -p 4567:4567 -v "${UDACITY_SOURCE}:/capstone" -v ${XSOCK}:${XSOCK} ${UDACITY_IMAGE} $@ 

fi
