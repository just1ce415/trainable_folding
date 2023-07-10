LOCAL_DIR="/Users/ernestglukhov/projects/folding/trainable_folding"
REMOTE_HOST="kikodze@129.49.83.196"
TARGET_DIR="/home/kikodze/projects/phospho/"
FROM_TO_ARG="$LOCAL_DIR $REMOTE_HOST:$TARGET_DIR"
echo "Uploading $FROM_TO_ARG"

COMMON_ARGS="-avz
--exclude=.idea
--exclude=.git"

rsync -e "ssh -p 0504" $COMMON_ARGS $FROM_TO_ARG