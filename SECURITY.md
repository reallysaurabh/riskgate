# Security policy

riskgate is a supervisory safety layer. Please report vulnerabilities privately by e-mail to **saurabhgupta3012@gmail.com** with the subject `riskgate security`. Include a description, a reproducer and the version. You will receive an acknowledgement within 5 working days.

## In scope
- Any way to make a gate return `CERTIFY` when it should not. Examples:
  - bypassing hard rules or fingerprint checks;
  - forging or tampering with a signed calibration artefact without detection;
  - prompt-injection paths into the certify path;
  - failure modes that do not fail closed.
- Information leaks through audit logs or telemetry beyond what the caller configured.

## Out of scope
- Incorrect consequence models, hypothesis providers or labels supplied by users (riskgate's guarantee is relative to them).
- Distribution shift. This is documented behaviour: the monitor detects it and the gate only tightens.

## Deployment hardening checklist
- Store calibration artefacts read-only and **sign** them (`Calibration.save(path, key=...)`, `load(..., require_signature=True)`).
- Bump `version` on every consequence-model or hypothesis-provider change, so that old calibrations become `VOID`.
- Run dry-runs against isolated copies: savepoints, staging namespaces, replicas.
- Use `AuditLog(redact=...)` for sensitive fields.
- Keep riskgate **in addition to**, never instead of, certified safety systems and access control.
