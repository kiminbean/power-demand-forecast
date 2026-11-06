# RE-BMS v6.0 Docker Deployment

## Current operation: v7 API container only (since 2026-11-06)

The running deployment uses `docker-compose.v7.yml` and starts only the `api` service.

```bash
# from the repo root
docker compose -f docker/docker-compose.v7.yml up -d --build --wait api
curl http://localhost:8506/health   # check models_loaded; /api/v1/health is healthy even without models
```

- Container `rebms-api` on :8506, `mem_limit: 1g`, Colima VM with 2 vCPU / 2 GB.
- The image installs `requirements-docker.txt`: CPU-only torch and versions pinned to the ones verified on the host. pandas stays below 3, because pandas 3 makes every `ar_gbm` prediction return 400.
- Demand models are mounted read-only: `models/production` (LSTM) and `models/demand_ar_gbm` (default `ar_gbm`). After retraining, restart the container; no rebuild is needed.
- The default demand model comes from `DEFAULT_MODEL` in the root `.env`. To roll back, set `DEFAULT_MODEL=demand_only` and run `up -d --force-recreate --wait api`. `docker restart` does not re-read `env_file`.
- Do not run `docker compose config` without `--quiet`: it prints the API key from the env file.

The v6 instructions below describe the older private deployment with Basic Auth.

## Private Deployment with Authentication

This setup deploys RE-BMS v6.0 with Basic Authentication for private access.

---

## Quick Start

### 1. Configure Authentication

```bash
cd docker

# Copy example env file
cp .env.example .env

# Edit with your credentials
nano .env
```

Set your credentials in `.env`:
```env
AUTH_USER=your_username
AUTH_PASS=your_secure_password
WEB_PORT=8600
```

### 2. Generate Authentication File

```bash
./setup-auth.sh
# Or with custom credentials:
./setup-auth.sh myuser mypassword
```

### 3. Build and Deploy

```bash
docker-compose -f docker-compose.v6.yml build
docker-compose -f docker-compose.v6.yml up -d
```

### 4. Access

Open browser: **http://localhost:8600**

Enter your username and password when prompted.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Docker Network                 │
│                                                  │
│  ┌──────────────┐        ┌──────────────┐       │
│  │   Nginx      │        │   FastAPI    │       │
│  │   (web)      │───────▶│   (api)      │       │
│  │   :80        │  /api  │   :8506      │       │
│  │              │        │              │       │
│  │  + Auth      │        │  + SMP Data  │       │
│  │  + Static    │        │  + ML Model  │       │
│  └──────┬───────┘        └──────────────┘       │
│         │                                        │
└─────────┼────────────────────────────────────────┘
          │
          ▼ :8600
    [ Browser with Login ]
```

---

## Services

| Service | Port | Description |
|---------|------|-------------|
| **web** | 8600 | Nginx + React (with auth) |
| **api** | 8506 | FastAPI backend |

---

## Commands

```bash
# Start services
docker-compose -f docker-compose.v6.yml up -d

# View logs
docker-compose -f docker-compose.v6.yml logs -f

# Stop services
docker-compose -f docker-compose.v6.yml down

# Rebuild after code changes
docker-compose -f docker-compose.v6.yml build --no-cache
docker-compose -f docker-compose.v6.yml up -d

# Check status
docker-compose -f docker-compose.v6.yml ps
```

---

## Security Options

### Option 1: Basic Auth (Default)
- Simple username/password
- Built into Nginx
- Good for personal use

### Option 2: IP Whitelist
Add to `nginx.conf`:
```nginx
# Allow only specific IPs
allow 192.168.1.0/24;
allow 10.0.0.0/8;
deny all;
```

### Option 3: VPN Only
- Deploy on private network
- Access only via VPN
- Most secure for enterprise

### Option 4: HTTPS with Certificates
```bash
# Generate self-signed cert (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout docker/ssl/key.pem \
  -out docker/ssl/cert.pem

# Update nginx.conf for SSL
```

---

## Troubleshooting

### Cannot access web
```bash
# Check if containers are running
docker ps

# Check web logs
docker logs rebms-web

# Check API health
curl http://localhost:8506/api/v1/health
```

### Authentication not working
```bash
# Regenerate htpasswd
./setup-auth.sh newuser newpassword

# Restart web container
docker-compose -f docker-compose.v6.yml restart web
```

### API connection failed
```bash
# Check API container
docker logs rebms-api

# Test API directly
curl http://localhost:8506/api/v1/health
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_PORT` | 8600 | Web server port |
| `AUTH_USER` | admin | Basic auth username |
| `AUTH_PASS` | rebms2025 | Basic auth password |
| `API_PORT` | 8506 | API server port |

---

## Data Volumes

| Volume | Container Path | Description |
|--------|----------------|-------------|
| `data/smp` | `/app/data/smp` | SMP historical data |
| `models` | `/app/models` | ML models |
| `models/production` (v7, read-only) | `/app/models/production` | Demand LSTM (`demand_only`, `weather_full`) |
| `models/demand_ar_gbm` (v7, read-only) | `/app/models/demand_ar_gbm` | Default demand model `ar_gbm` |

---

## Production Checklist

- [ ] Change default credentials in `.env`
- [ ] Use HTTPS in production
- [ ] Set up proper firewall rules
- [ ] Configure backup for data volumes
- [ ] Set up monitoring and alerts
- [ ] Consider using Docker secrets for credentials
