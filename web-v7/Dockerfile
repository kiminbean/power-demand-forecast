# RE-BMS v6.0 - React Desktop Web Application
# Multi-stage build for production

# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build production
RUN npm run build

# Stage 2: Production (Nginx)
FROM nginx:alpine AS production

# Install apache2-utils for htpasswd
RUN apk add --no-cache apache2-utils

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create .htpasswd placeholder (will be overwritten by docker-compose)
RUN touch /etc/nginx/.htpasswd

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
