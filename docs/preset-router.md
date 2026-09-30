# Model IDs that load different serving presets

`serve/preset_router.py` is an optional local HTTP router for clients that require unique model IDs. It advertises all configured IDs at `/v1/models` and uses the `model` field in OpenAI `/v1/chat/completions` or Anthropic `/v1/messages` requests to select a server configuration. Selecting an entry in a client alone does not trigger a reload; the first inference request does.

Only one backend server is resident. Switching waits for active HTTP requests to finish, stops the old server process tree, and launches the selected preset. This reload typically takes tens of seconds; clients must allow time for it before receiving response headers. Requests are admitted in arrival order, so a queued switch cannot be indefinitely delayed by later requests for the old preset. Requests for the active preset can proceed concurrently; the backend enforces its configured inference concurrency.

Run `python serve/preset_router.py --config <router-config.json>`. The router binds loopback port 8080 by default; the backend uses loopback port 8081. Both must initially be free. A local Windows installation can use `run-presets.bat`. Keep the router running instead of launching individual preset servers.

Example router config:

```json
{
  "port": 8080,
  "backend_port": 8081,
  "python": "C:/path/to/python.exe",
  "log_dir": "C:/path/to/preset-logs",
  "models": {
    "model-c1-vision": "C:/path/to/vision-server.json",
    "model-c1": "C:/path/to/text-server.json",
    "model-c4": "C:/path/to/concurrent-server.json"
  }
}
```

Each referenced file is a normal Strata server config. Its `model_name` must equal the corresponding router ID. Model weights can be shared on disk. The router does not alter context, cache, vision, sampling or concurrency settings. It does not enable vision in concurrent mode.

`/router/status` reports the loaded ID, active/queued request counts and whether a switch is in progress. `/props?model=<id>` provides preset context and modality metadata without loading weights. `/health` checks the router itself. The router does not proxy the Strata web interface or all management endpoints; the active backend's web interface is available at port 8081, but requests should go through the router to participate in switching coordination. This is a loopback-only local service, not an authenticated public gateway.

Switching presets during work in several chats is supported but requires reloads when their requested presets differ. Four subagents using the same c=4 ID share the loaded c=4 backend. Ctrl+C on the router shuts down its backend. On Windows, forced termination should target the router's entire process tree.

Lightweight tests in `tests/concurrency/test_preset_router.py` cover concurrent leases, switch ordering, failed-start recovery, unique model listing and SSE forwarding without loading a model.
