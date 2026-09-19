# Runtime source

The public entry points are in `scripts/`. The modules in this directory contain the audited experiment runtime used to reproduce the final paper configurations.

`eventsleep2_timestamp_protocol_v1.py` is intentionally retained byte-for-byte from the audited experiment because the adapter source SHA-256 is part of the EventSleep2 timestamp dataset fingerprint. The numerical version constants in the runtime are provenance identifiers; they are not alternative models exposed by this release.
