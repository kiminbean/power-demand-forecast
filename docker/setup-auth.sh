#!/bin/bash
#
# RE-BMS v6.0 Authentication Setup
# Creates htpasswd file for Basic Authentication
#
# Usage:
#   ./setup-auth.sh [username] [password]
#   ./setup-auth.sh  # Uses .env file or defaults
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTPASSWD_FILE="$SCRIPT_DIR/htpasswd"

# Load .env if exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

# Get credentials
USERNAME="${1:-${AUTH_USER:-admin}}"
PASSWORD="${2:-${AUTH_PASS:-rebms2025}}"

echo ""
echo "=================================================="
echo " RE-BMS v6.0 Authentication Setup"
echo "=================================================="
echo ""

# Check for htpasswd command
if command -v htpasswd &> /dev/null; then
    # Use Apache htpasswd
    htpasswd -bc "$HTPASSWD_FILE" "$USERNAME" "$PASSWORD"
elif command -v openssl &> /dev/null; then
    # Use OpenSSL fallback
    HASH=$(openssl passwd -apr1 "$PASSWORD")
    echo "$USERNAME:$HASH" > "$HTPASSWD_FILE"
else
    echo "Error: Neither htpasswd nor openssl found."
    echo "Install apache2-utils or openssl."
    exit 1
fi

chmod 600 "$HTPASSWD_FILE"

echo "Authentication configured:"
echo "  Username: $USERNAME"
echo "  Password: ********"
echo "  File: $HTPASSWD_FILE"
echo ""
echo "To deploy:"
echo "  cd docker"
echo "  docker-compose -f docker-compose.v6.yml up -d"
echo ""
echo "Access at: http://localhost:${WEB_PORT:-8600}"
echo ""
