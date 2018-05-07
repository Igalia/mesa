if [ $# != 5 ]; then
    echo "Usage: $0 <username> <password> <image> <tag> <property>"
    exit 0
fi

USERNAME=$1
PASSWORD=$2
IMAGE=$3
TAG=$4
PROPERTY=$5

AUTHPOINT=$(curl -L -D - -s https://${CI_REGISTRY}/v2/ | grep Www-Authenticate)

REALM=$(echo -n ${AUTHPOINT} | cut -d \" -f 2)
SERVICE=$(echo -n ${AUTHPOINT} | cut -d \" -f 4)

TOKEN=$(curl -L -s --user "${USERNAME}:${PASSWORD}" "${REALM}?client_id=docker&offline_token=true&service=${SERVICE}&scope=repository:${IMAGE}:pull" | jq -r ".token")

DIGEST=$(curl -L -s -H "Accept: application/vnd.docker.distribution.manifest.v2+json" -H "Authorization: Bearer ${TOKEN}" "https://${CI_REGISTRY}/v2/${IMAGE}/manifests/${TAG}" | jq -r ".config.digest")

curl -L -s -H "Authorization: Bearer ${TOKEN}" "https://${CI_REGISTRY}/v2/${IMAGE}/blobs/${DIGEST}" | jq -r "${PROPERTY}"
