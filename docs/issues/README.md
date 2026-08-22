# Issue Import Files

This directory contains one markdown file per issue-ready ticket.

> These files are April 2026 import snapshots. Revalidate their status and
> dependencies against the canonical [product roadmap](../Roadmap.md),
> [feature index](../features/README.md), and
> [technical-debt register](../technical-debt/README.md) before import.

## Suggested Bulk Import Command (GitHub CLI)

```bash
for f in docs/issues/P*.md; do
  title="$(sed -n '1s/^# //p' "$f")"
  labels="$(sed -n 's/^Labels: //p' "$f")"
  tmp_body="$(mktemp)"
  awk 'NR==1 {next} /^Priority: / {next} /^Owner: / {next} /^Effort: / {next} /^Risk: / {next} /^Dependencies: / {next} /^Labels: / {next} {print}' "$f" > "$tmp_body"

  cmd=(gh issue create --title "$title" --body-file "$tmp_body")
  IFS=',' read -ra parts <<< "$labels"
  for lbl in "${parts[@]}"; do
    clean="$(echo "$lbl" | xargs)"
    if [ -n "$clean" ]; then
      cmd+=(--label "$clean")
    fi
  done

  echo "Creating issue from $f"
  "${cmd[@]}"
  rm -f "$tmp_body"
done
```

Optionally add `--repo <owner/repo>` to the `gh issue create` command inside the loop.
