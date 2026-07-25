#!/bin/bash
# install_geoserver.sh
# GeoServer installation and configuration for WSL2 Ubuntu 24.04
# BDA thesis pipeline - one-time setup, idempotent
# Usage: bash install_geoserver.sh
# Place in: J:\GoogleDrive\masterthesis\notebooks\

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

GEOSERVER_VERSION="2.28.2"
INSTALL_DIR="/opt/geoserver"
DATA_DIR="/opt/geoserver_data"
LOG_DIR="/var/log/geoserver"
GEOSERVER_PORT="8080"

# Passwords - change before running
ADMIN_PASS="bda_admin_2026"
VIEWER_PASS="bda_viewer_2026"
VIEWER_USER="viewer"

# JVM - tuned for 128GB WSL machine, safe for demo VM with less RAM
JVM_XMS="1g"
JVM_XMX="4g"

BASE_URL="https://sourceforge.net/projects/geoserver/files/GeoServer/${GEOSERVER_VERSION}"
EXT_URL="https://sourceforge.net/projects/geoserver/files/GeoServer/${GEOSERVER_VERSION}/extensions"

# =============================================================================
# HELPERS
# =============================================================================

info() { echo "[INFO] $1"; }
ok()   { echo "[OK]   $1"; }
warn() { echo "[WARN] $1"; }

wait_for_geoserver() {
    info "Waiting for GeoServer to start on port ${GEOSERVER_PORT}..."
    for i in $(seq 1 60); do
        if curl -s -o /dev/null "http://localhost:${GEOSERVER_PORT}/geoserver/web/"; then
            ok "GeoServer is up"
            return 0
        fi
        sleep 3
    done
    echo "[ERROR] GeoServer did not start within 180s"
    exit 1
}

rest() {
    # rest METHOD PATH [BODY] [CONTENT_TYPE]
    local method="$1"
    local path="$2"
    local body="${3:-}"
    local ctype="${4:-application/json}"
    if [ -n "$body" ]; then
        curl -s -u "admin:${ADMIN_PASS}" -X "$method" \
            -H "Content-Type: ${ctype}" \
            -d "$body" \
            "http://localhost:${GEOSERVER_PORT}/geoserver/rest/${path}"
    else
        curl -s -u "admin:${ADMIN_PASS}" -X "$method" \
            "http://localhost:${GEOSERVER_PORT}/geoserver/rest/${path}"
    fi
}

# =============================================================================
# STEP 0 - WSL systemd check
# =============================================================================

info "Checking WSL systemd..."
if ! grep -q "systemd=true" /etc/wsl.conf 2>/dev/null; then
    warn "systemd not enabled in /etc/wsl.conf - adding it"
    if ! grep -q "\[boot\]" /etc/wsl.conf 2>/dev/null; then
        echo -e "\n[boot]\nsystemd=true" | sudo tee -a /etc/wsl.conf > /dev/null
    else
        sudo sed -i '/\[boot\]/a systemd=true' /etc/wsl.conf
    fi
    warn "WSL restart required for systemd. Run: wsl --shutdown, then restart WSL and re-run this script."
    warn "Continuing without systemd service registration for now."
    SYSTEMD_OK=false
else
    ok "systemd=true already set"
    SYSTEMD_OK=true
fi

# =============================================================================
# STEP 1 - Java 17
# =============================================================================

info "Installing OpenJDK 17..."
sudo apt-get update -qq
sudo apt-get install -y -qq openjdk-17-jdk unzip curl

JAVA_HOME_PATH=$(dirname $(dirname $(readlink -f $(which java))))
ok "Java: $(java -version 2>&1 | head -1)"

# =============================================================================
# STEP 2 - Download and install GeoServer
# =============================================================================

if [ -d "$INSTALL_DIR" ]; then
    ok "GeoServer already installed at $INSTALL_DIR - skipping download"
else
    info "Downloading GeoServer ${GEOSERVER_VERSION}..."
    TMP_DIR=$(mktemp -d)
    cd "$TMP_DIR"
    wget -q --show-progress \
        "${BASE_URL}/geoserver-${GEOSERVER_VERSION}-bin.zip" \
        -O geoserver.zip
    sudo unzip -q geoserver.zip -d /opt/
    sudo mv "/opt/geoserver-${GEOSERVER_VERSION}" "$INSTALL_DIR"
    rm -rf "$TMP_DIR"
    ok "GeoServer installed at $INSTALL_DIR"
fi

# =============================================================================
# STEP 3 - Data directory and log directory
# =============================================================================

sudo mkdir -p "$DATA_DIR" "$LOG_DIR"
sudo chown -R "$USER":"$USER" "$DATA_DIR" "$LOG_DIR"
sudo chown -R "$USER":"$USER" "$INSTALL_DIR"

# Copy default data dir if empty
if [ ! -f "$DATA_DIR/global.xml" ]; then
    cp -r "$INSTALL_DIR/data_dir/." "$DATA_DIR/"
    ok "Data directory initialized at $DATA_DIR"
else
    ok "Data directory already exists at $DATA_DIR"
fi

# =============================================================================
# STEP 4 - JVM and startup configuration
# =============================================================================

info "Configuring JVM options..."

cat > "$INSTALL_DIR/bin/setenv.sh" << EOF
#!/bin/bash
export JAVA_HOME=${JAVA_HOME_PATH}
export GEOSERVER_DATA_DIR=${DATA_DIR}
export GEOSERVER_HOME=${INSTALL_DIR}
export JAVA_OPTS="-Xms${JVM_XMS} -Xmx${JVM_XMX} -XX:+UseG1GC -XX:MaxGCPauseMillis=200 -Dfile.encoding=UTF-8 -Djavax.servlet.request.encoding=UTF-8 -Djavax.servlet.response.encoding=UTF-8 -server -Duser.timezone=UTC"
# COG (Cloud Optimized GeoTIFF) native support
export JAVA_OPTS="\$JAVA_OPTS -Dgeoserver.login.autocomplete=off"
# Disable CSRF for REST API access from Python scripts on localhost
export JAVA_OPTS="\$JAVA_OPTS -DGEOSERVER_CSRF_DISABLED=true"
EOF

chmod +x "$INSTALL_DIR/bin/setenv.sh"
ok "JVM options configured"

# =============================================================================
# STEP 5 - Download and install extensions
# =============================================================================

EXT_LIB_DIR="$INSTALL_DIR/webapps/geoserver/WEB-INF/lib"
TMP_EXT=$(mktemp -d)

install_extension() {
    local name="$1"
    local zipname="geoserver-${GEOSERVER_VERSION}-${name}-plugin.zip"
    local marker="$INSTALL_DIR/.ext_${name}_installed"
    if [ -f "$marker" ]; then
        ok "Extension ${name} already installed"
        return
    fi
    info "Downloading extension: ${name}..."
    wget -q --show-progress \
        "${EXT_URL}/${zipname}" \
        -O "$TMP_EXT/${zipname}"
    unzip -q -o "$TMP_EXT/${zipname}" -d "$TMP_EXT/${name}/"
    cp "$TMP_EXT/${name}/"*.jar "$EXT_LIB_DIR/"
    touch "$marker"
    ok "Extension ${name} installed"
}

# importer: REST bulk import, essential for Python pipeline automation
install_extension "importer"

# css: author styles in CSS syntax instead of XML SLD (optional but saves time)
install_extension "css"

# control-flow: rate limiting and request queue management
install_extension "control-flow"

# monitor: request logging to disk, useful for debugging and demo metrics
install_extension "monitor"

rm -rf "$TMP_EXT"

# =============================================================================
# STEP 6 - control-flow configuration
# =============================================================================

CTRLFLOW_CONF="$DATA_DIR/controlflow.properties"
if [ ! -f "$CTRLFLOW_CONF" ]; then
    cat > "$CTRLFLOW_CONF" << EOF
# Max concurrent WMS requests total
ows.global=20
# Max concurrent requests per IP
ows.user=10
# Max WMS GetMap queue
wms.getmap=16
# Timeout in seconds
timeout=60
EOF
    ok "control-flow configured"
fi

# =============================================================================
# STEP 7 - systemd service
# =============================================================================

SERVICE_FILE="/etc/systemd/system/geoserver.service"

if $SYSTEMD_OK; then
    if [ ! -f "$SERVICE_FILE" ]; then
        info "Creating systemd service..."
        sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=GeoServer WMS
After=network.target

[Service]
Type=simple
User=${USER}
Environment=JAVA_HOME=${JAVA_HOME_PATH}
Environment=GEOSERVER_DATA_DIR=${DATA_DIR}
Environment=GEOSERVER_HOME=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/bin/startup.sh
ExecStop=${INSTALL_DIR}/bin/shutdown.sh
StandardOutput=append:${LOG_DIR}/geoserver.log
StandardError=append:${LOG_DIR}/geoserver-error.log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        sudo systemctl daemon-reload
        sudo systemctl enable geoserver
        ok "systemd service created and enabled"
    else
        ok "systemd service already exists"
    fi

    info "Starting GeoServer service..."
    sudo systemctl start geoserver || true
else
    warn "Skipping systemd service registration (systemd not yet active)"
    warn "Starting GeoServer manually for configuration..."
    source "$INSTALL_DIR/bin/setenv.sh"
    nohup "$INSTALL_DIR/bin/startup.sh" > "$LOG_DIR/geoserver.log" 2>&1 &
    echo $! > /tmp/geoserver.pid
    ok "GeoServer started manually (PID: $(cat /tmp/geoserver.pid))"
fi

# =============================================================================
# STEP 8 - Wait and configure via REST
# =============================================================================

wait_for_geoserver
sleep 5

# Change admin password (default is admin/geoserver)
info "Changing admin password..."
# First call uses default password
curl -s -u "admin:geoserver" -X PUT \
    -H "Content-Type: application/json" \
    -d "{\"oldPassword\":\"geoserver\",\"newPassword\":\"${ADMIN_PASS}\"}" \
    "http://localhost:${GEOSERVER_PORT}/geoserver/rest/security/self/password" || \
    ok "Admin password already changed (skipping)"

# Verify new password works
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -u "admin:${ADMIN_PASS}" \
    "http://localhost:${GEOSERVER_PORT}/geoserver/rest/about/version.json")
if [ "$HTTP_CODE" != "200" ]; then
    echo "[ERROR] Could not authenticate with new admin password. Already changed to something else?"
    exit 1
fi
ok "Admin password set"

# =============================================================================
# STEP 9 - Create viewer (read-only) user
# =============================================================================

info "Creating viewer user..."

# Create user
rest PUT "security/usergroup/users/${VIEWER_USER}" \
    "{\"org.geoserver.rest.security.xml.JaxbUser\":{\"userName\":\"${VIEWER_USER}\",\"password\":\"${VIEWER_PASS}\",\"enabled\":true}}"

# Assign ROLE_AUTHENTICATED (WMS GetMap + GetCapabilities, no admin)
rest POST "security/roles/role/ROLE_VIEWER" "" || true
rest POST "security/roles/role/ROLE_VIEWER/user/${VIEWER_USER}" "" || true

ok "Viewer user '${VIEWER_USER}' created"

# =============================================================================
# STEP 10 - Create bda_ukraine workspace
# =============================================================================

info "Creating bda_ukraine workspace..."
rest POST "workspaces" \
    '{"workspace":{"name":"bda_ukraine"}}'
ok "Workspace bda_ukraine created"

# =============================================================================
# STEP 11 - Cascaded OSM WMS store (local demo basemap)
# =============================================================================

info "Adding cascaded OSM WMS store (local demo only)..."
rest POST "workspaces/bda_ukraine/wmsstores" \
    '{"wmsStore":{"name":"osm_mundialis","type":"WMS","enabled":true,"capabilitiesURL":"https://ows.mundialis.de/services/service?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities","maxConnections":6,"readTimeout":60,"connectTimeout":30}}'
ok "OSM WMS store added"

# =============================================================================
# STEP 12 - Global WMS service settings
# =============================================================================

info "Configuring global WMS settings..."
rest PUT "services/wms/settings" \
    '{"wms":{"enabled":true,"name":"WMS","title":"BDA Ukraine WMS","maintainer":"Marco Heinzen","abstract":"Battlefield Damage Assessment - Ukraine conflict cities","fees":"NONE","accessConstraints":"NONE","versions":{"org.geoserver.util.Version":[{"version":"1.1.1"},{"version":"1.3.0"}]},"citeCompliant":false,"onlineResource":"http://localhost:8080/geoserver","schemaBaseURL":"http://schemas.opengis.net","verbose":false,"maxRequestMemory":65536,"maxRenderingTime":120,"maxRenderingErrors":100,"globalWatermarking":false,"watermarkTransparency":100,"watermarkPosition":"BOT_RIGHT","interpolation":"Nearest","getMapMimeTypeCheckingEnabled":false,"dynamicStylingDisabled":false}}'
ok "WMS settings configured"

# =============================================================================
# DONE
# =============================================================================

echo ""
echo "============================================================"
echo "GeoServer ${GEOSERVER_VERSION} installation complete"
echo "============================================================"
echo ""
echo "  URL:          http://localhost:${GEOSERVER_PORT}/geoserver"
echo "  Admin:        admin / ${ADMIN_PASS}"
echo "  Viewer:       ${VIEWER_USER} / ${VIEWER_PASS}"
echo "  Install dir:  ${INSTALL_DIR}"
echo "  Data dir:     ${DATA_DIR}"
echo "  Log dir:      ${LOG_DIR}"
echo "  Workspace:    bda_ukraine"
echo ""
echo "  Extensions installed:"
echo "    - importer    (REST bulk import for Python pipeline)"
echo "    - css         (CSS style authoring)"
echo "    - control-flow (rate limiting)"
echo "    - monitor     (request logging)"
echo ""
if $SYSTEMD_OK; then
    echo "  Service management:"
    echo "    sudo systemctl start|stop|restart|status geoserver"
    echo "    Logs: journalctl -u geoserver -f"
    echo "          tail -f ${LOG_DIR}/geoserver.log"
else
    echo "  IMPORTANT: Restart WSL (wsl --shutdown) to activate systemd,"
    echo "  then re-run this script to register the autostart service."
    echo "  Manual start: source ${INSTALL_DIR}/bin/setenv.sh && ${INSTALL_DIR}/bin/startup.sh"
fi
echo ""
echo "  Next step: run export_results.py to push city layers"
echo "============================================================"
