# ─────────────────────────────────────────────────────────────────────────────
# PSIP Dashboard — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder) — installs deps and runs `vite build`
#   Stage 2 (runtime) — nginx:alpine serves the static dist/ folder
#
# The Vite dev-proxy (/api → localhost:8000) is replaced in production by
# the nginx upstream block below, which forwards /api/ to the psip-api service.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --silent

COPY . .
RUN npm run build

# ── Stage 2: Serve ────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Remove default nginx site
RUN rm /etc/nginx/conf.d/default.conf

# Custom nginx config: serve React app + proxy /api/ to the API container
COPY --from=builder /app/nginx.conf /etc/nginx/conf.d/psip.conf

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
