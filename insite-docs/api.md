# API

Most users do not need the API. The WebApp handles all normal tasks.

If your team uses integrations, these are the common endpoints:
- `GET /api/queue` - current queue
- `GET /api/history` - recent runs with downloads
- `POST /api/upload` - upload ZIPs with options
- `DELETE /api/run/:id` - delete a run and its files
- `GET/POST/DELETE /api/autochar` - manage presets
- `GET /health` - health check
