cd ~/pv_forecast_30d

cat > backup_repo.sh << 'EOF'
#!/usr/bin/env bash
set -e

# Use current directory as repo root. You must run this script from the repo root.
REPO_ROOT="$(pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

BACKUP_DIR="${REPO_ROOT}/_backups"
mkdir -p "${BACKUP_DIR}"

ARCHIVE_NAME="${REPO_NAME}_backup_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${BACKUP_DIR}/${ARCHIVE_NAME}"

echo "Backing up repo:"
echo "  From: ${REPO_ROOT}"
echo "  To:   ${ARCHIVE_PATH}"
echo

tar czf "${ARCHIVE_PATH}" \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='_backups' \
  .

echo
echo "Backup created successfully:"
ls -lh "${ARCHIVE_PATH}"
EOF
