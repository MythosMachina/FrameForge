# Notifications (Email, Discord, Slack, Webhook)

Notifications let you know when a run finishes, fails, or when the queue is empty.

## Who can set this up?
If you can edit Settings in the UI, you can configure notifications. If not, ask an admin.

## What you can notify
- Job finished
- Job failed
- Queue finished (everything is idle)

## Channels (what they mean)
- Email: sends a summary to an email address.
- Discord: sends a message to a Discord channel.
- Slack: sends a message to a Slack channel.
- Webhook: sends a JSON payload to any URL you control.

## Quick setup (recommended order)
1) Open Settings.
2) Turn ON `notifications_enabled`.
3) Turn ON one or more channels.
4) Turn ON the event types you want (job finish, job failed, queue finish).
5) Save.

## Discord webhook setup
1) In Discord, go to your server and channel.
2) Channel Settings → Integrations → Webhooks → New Webhook.
3) Copy the Webhook URL.
4) In Settings, paste it into `discord_webhook_url`.
5) Enable `notify_channel_discord` and the events you want.

## Slack webhook setup
1) In Slack, create an Incoming Webhook for your workspace.
2) Copy the Webhook URL.
3) In Settings, paste it into `slack_webhook_url`.
4) Enable `notify_channel_slack` and the events you want.

## Webhook (custom) setup
Use this if you want your own system to receive JSON.
1) Create an HTTP endpoint that accepts POST.
2) Paste the URL into `webhook_url`.
3) (Optional) set `webhook_secret` for signed requests.
4) Enable `notify_channel_webhook` and the events you want.

Webhook payload example (fields you will receive):
- type
- run_id
- run_name
- status
- last_step
- error
- created_at
- started_at
- finished_at
- dataset_url
- lora_url
- instance_label
- instance_url

## Email setup (simple)
Fill these fields in Settings:
- smtp_host
- smtp_port
- smtp_user (if required)
- smtp_pass (if required)
- smtp_from
- smtp_to
- smtp_tls or smtp_ssl

Then enable:
- notify_channel_email
- notify_job_finish / notify_job_failed / notify_queue_finish

## Troubleshooting
- If nothing sends, double-check `notifications_enabled`.
- If only some channels work, check the channel-specific URL fields.
- If emails fail, verify SMTP settings and credentials.
