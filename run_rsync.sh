LOCAL_DIR="/home/kikodze/projects/mhc/trainable_folding"
REMOTE_HOST="eglukhov@enki.ams.stonybrook.edu"
TARGET_DIR="/home/eglukhov/projects/mhc"
FROM_TO_ARG="$LOCAL_DIR $REMOTE_HOST:$TARGET_DIR"
echo "Uploading $FROM_TO_ARG"

COMMON_ARGS="-avz
--exclude=.idea
--exclude=.git"

rsync $COMMON_ARGS $FROM_TO_ARG