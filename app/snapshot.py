"""Fulfillment status snapshots for immutable release batches.

A snapshot combines the frozen batch manifest with:

* an obligation schedule (offset/unit/depends_on rules);
* evidence materials carrying their own ``provided_at`` timestamps;
* nested ``all``/``any`` evidence rules.

The batch manifest has already deduplicated shared components, so each
merged ``(component, obligation)`` pair is evaluated exactly once.  A
material participates in an instance only when its local date in the
requested IANA timezone is no later than both the current date and the
obligation's final deadline date.  Satisfied instances may be evidenced
jointly by several materials; product-scoped materials cover every
component, while component-scoped materials cover only their declared
component.

All semantic validation happens before the result is assembled.  An
invalid schedule or evidence payload therefore fails as a whole with
HTTP 422 -- it never produces a partial snapshot.
"""

from datetime import timedelta, timezone

from .common import ClientError, canonical, stable_hash
from .fulfillment import (
    _SCOPES,
    UnprocessableError,
    _iter_leaf_indexes,
    _missing_leaves,
    _witness,
    build_formula,
    eval_circuit,
    normalize_rules,
)
from .schedule import (
    _add_business_days,
    _parse_rfc3339,
    _tighten,
    normalize_schedule_request,
)


def _utc(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_snapshot_context(body, required_observations):
    """Validate timestamps and reuse the schedule request normalizer.

    The assessment scheduler requires rules for artifact obligations.  A
    batch snapshot is evaluated over deduplicated component obligations, so
    pass that union as the required schedule vocabulary here.
    """
    if not isinstance(body, dict):
        raise ClientError("request body must be a JSON object")
    current_at = _parse_rfc3339(body.get("current_at"), "current_at")
    normalized = normalize_schedule_request(body, required_observations)
    release_at = normalized[0]
    return (current_at,) + normalized


def normalize_snapshot_materials(body, tz, component_ids, current_date):
    """Validate and deduplicate timestamped evidence materials.

    Returns ``(materials_by_id, sorted_material_records, duplicates)``.
    Equal digests with conflicting identifying metadata -- including the
    evidence timestamp -- are a 422.  Repeated digests with identical
    metadata collapse onto the smallest material id, matching the existing
    fulfillment-check semantics while remaining independent of array order.
"""
    raw_list = body.get("materials", [])
    if not isinstance(raw_list, list):
        raise ClientError("materials must be an array")

    entries = []
    seen_ids = set()
    for i, raw in enumerate(raw_list):
        where = "materials[%d]" % i
        if not isinstance(raw, dict):
            raise ClientError("%s must be an object" % where)

        def required_str(key):
            val = raw.get(key)
            if not isinstance(val, str) or not val.strip():
                raise ClientError("%s.%s must be a non-empty string" % (where, key))
            return val.strip()

        mid = required_str("id")
        if mid in seen_ids:
            raise ClientError("duplicate material id %r" % mid)
        seen_ids.add(mid)
        mtype = required_str("type")
        scope = required_str("scope")
        if scope not in _SCOPES:
            raise ClientError("%s.scope must be one of %s" % (where, list(_SCOPES)))
        component = None
        if scope == "component":
            component = required_str("component")
            if component not in component_ids:
                raise UnprocessableError(
                    "%s.component references unknown component %r" % (where, component)
                )
        elif raw.get("component") is not None:
            raise ClientError(
                "%s: product-scoped materials must not declare a component" % where
            )
        provided_at = _parse_rfc3339(raw.get("provided_at"), "%s.provided_at" % where)
        provided_date = provided_at.astimezone(tz).date()
        if provided_date > current_date:
            raise UnprocessableError(
                "%s.provided_at %r is later than the requested current date %r"
                % (where, raw.get("provided_at"), current_date.isoformat())
            )
        digest = raw.get("digest")
        if digest is not None:
            if not isinstance(digest, str) or not digest.strip():
                raise ClientError("%s.digest must be a non-empty string" % where)
            digest = digest.strip()
        entries.append(
            {
                "id": mid,
                "digest": digest,
                "type": mtype,
                "scope": scope,
                "component": component,
                "provided_at": provided_at,
                "provided_date": provided_date,
            }
        )

    groups = {}
    for entry in entries:
        if entry["digest"] is None:
            continue
        groups.setdefault(entry["digest"], []).append(entry)

    materials = {entry["id"]: entry for entry in entries}
    duplicates = []
    for digest in sorted(groups):
        members = groups[digest]
        signatures = {
            stable_hash(
                {
                    "type": m["type"],
                    "scope": m["scope"],
                    "component": m["component"],
                    "provided_date": m["provided_date"].isoformat(),
                }
            )
            for m in members
        }
        if len(signatures) > 1:
            details = sorted(
                "%s(type=%s, scope=%s, component=%s)"
                % (m["id"], m["type"], m["scope"], m["component"])
                for m in members
            )
            raise UnprocessableError(
                "digest %s submitted with conflicting metadata: %s"
                % (digest, "; ".join(details))
            )
        if len(members) == 1:
            continue
        canonical_id = min(m["id"] for m in members)
        duplicates.append(
            {
                "digest": digest,
                "canonical_id": canonical_id,
                "duplicate_ids": sorted(m["id"] for m in members),
            }
        )
        for m in members:
            if m["id"] != canonical_id:
                del materials[m["id"]]

    records = []
    for mid in sorted(materials):
        m = materials[mid]
        records.append(
            {
                "id": mid,
                "digest": m["digest"],
                "type": m["type"],
                "scope": m["scope"],
                "component": m["component"],
                "provided_at": _utc(m["provided_at"]),
                "provided_date": m["provided_date"].isoformat(),
            }
        )
    return materials, records, duplicates


def _deadlines(rules, release_date, workdays, holidays):
    raw = {}
    for ob, rule in rules.items():
        if rule["unit"] == "calendar_day":
            raw[ob] = release_date + timedelta(days=rule["offset"])
        else:
            raw[ob] = _add_business_days(
                release_date, rule["offset"], workdays, holidays
            )
    final, chains = _tighten(rules, raw)
    return raw, final, chains


def _formula_leaf_materials(formula, material_ids, materials, instance, eligible):
    """Return per-leaf sorted material ids, restricted to eligible ids."""
    if formula is None:
        return []
    _tree, leaf_masks, leaves, _gates, _root, _nleaves = formula
    out = []
    for li in range(len(leaves)):
        ids = []
        for j, mid in enumerate(material_ids):
            if mid in eligible and ((leaf_masks[j] >> li) & 1):
                ids.append(mid)
        out.append(tuple(sorted(ids)))
    return out


def _evaluate_instance(instance, rule_node, materials, material_ids, deadline_date,
                       current_date):
    formula = build_formula(
        materials, material_ids, rule_node, instance
    )
    if formula is None:
        # Validation makes this unreachable for required obligations; keep
        # the guard explicit so a future change cannot emit a partial item.
        raise UnprocessableError(
            "missing evidence rule for obligation %r at component %r"
            % (instance["obligation"], instance["component"])
        )

    tree, leaf_masks, leaves, gates, root, nleaves = formula
    cutoff = min(deadline_date, current_date)
    eligible = {
        mid
        for mid in material_ids
        if materials[mid]["provided_date"] <= cutoff
    }
    truth = 0
    for j, mid in enumerate(material_ids):
        if mid in eligible:
            truth |= leaf_masks[j]
    satisfied = eval_circuit(gates, root, truth, nleaves)
    eligible_leaf_mats = _formula_leaf_materials(
        formula, material_ids, materials, instance, eligible
    )

    if satisfied:
        matched = sorted(_witness(tree, eligible_leaf_mats, eligible))
        return "satisfied", None, matched

    matched_any_leaf = any(
        eligible_leaf_mats[li] for li in _iter_leaf_indexes(tree)
    )
    candidate = sorted({
        mid
        for li in _iter_leaf_indexes(tree)
        for mid in eligible_leaf_mats[li]
    })
    if matched_any_leaf:
        missing_indexes = _missing_leaves(tree, eligible_leaf_mats)
        missing_leaves = sorted(
            (leaves[li] for li in missing_indexes), key=canonical
        )
        reason = "incomplete-evidence"
    else:
        missing_leaves = []
        reason = "no-matching-material"
    gap = {
        "reason": reason,
        "candidate_materials": candidate,
        "missing_leaves": missing_leaves,
    }
    status = "overdue" if current_date > deadline_date else "pending"
    return status, gap, []


def build_batch_snapshot(body, manifest):
    component_ids = sorted(c["id"] for c in manifest.get("components", []))
    required_observations = sorted({
        ob
        for component in manifest.get("components", [])
        for ob in component.get("obligations", [])
    })

    (
        current_at,
        release_at,
        tz_name,
        tz,
        workdays,
        holidays,
        rules,
    ) = normalize_snapshot_context(body, required_observations)

    try:
        evidence = normalize_rules(body, component_ids)
    except ClientError as exc:
        # A structurally valid rule that points outside the frozen batch is
        # semantically unprocessable for this snapshot, rather than a generic
        # malformed-request error.
        if "references unknown component" in str(exc):
            raise UnprocessableError(str(exc))
        raise
    evidence_by_obligation = evidence["obligations"]
    missing_evidence = sorted(
        ob for ob in required_observations if ob not in evidence_by_obligation
    )
    if missing_evidence:
        raise UnprocessableError(
            "missing evidence rules for component obligations: %s"
            % ", ".join(missing_evidence)
        )

    release_date = release_at.astimezone(tz).date()
    current_date = current_at.astimezone(tz).date()
    materials, material_records, duplicates = normalize_snapshot_materials(
        body, tz, set(component_ids), current_date
    )

    raw, final, chains = _deadlines(rules, release_date, workdays, holidays)

    components_by_obligation = {}
    for component in manifest.get("components", []):
        for ob in component.get("obligations", []):
            components_by_obligation.setdefault(ob, set()).add(component["id"])

    schedule = []
    for ob in sorted(rules):
        rule = rules[ob]
        schedule.append(
            {
                "obligation": ob,
                "components": sorted(components_by_obligation.get(ob, set())),
                "offset": rule["offset"],
                "unit": rule["unit"],
                "depends_on": rule["depends_on"],
                "raw_deadline": raw[ob].isoformat(),
                "deadline": final[ob].isoformat(),
                "tightened": final[ob] != raw[ob],
                "tightening_chain": chains[ob] if final[ob] != raw[ob] else [],
            }
        )

    material_ids = sorted(materials)
    items = []
    item_statuses = {}
    counts = {"satisfied": 0, "pending": 0, "overdue": 0}
    for component in sorted(manifest.get("components", []), key=lambda c: c["id"]):
        for ob in sorted(component.get("obligations", [])):
            instance = {"component": component["id"], "obligation": ob}
            status, gap, matched = _evaluate_instance(
                instance,
                evidence_by_obligation[ob],
                materials,
                material_ids,
                final[ob],
                current_date,
            )
            counts[status] += 1
            item_id = "component:%s:%s" % (component["id"], ob)
            item_statuses[item_id] = status
            items.append(
                {
                    "id": item_id,
                    "component": component["id"],
                    "obligation": ob,
                    "deadline": final[ob].isoformat(),
                    "status": status,
                    "gap": gap,
                    "matched_materials": matched,
                }
            )

    out = {
        "batch_id": manifest["batch_id"],
        "batch_hash": manifest["batch_hash"],
        "current_at": _utc(current_at),
        "release_at": _utc(release_at),
        "timezone": tz_name,
        "current_date": current_date.isoformat(),
        "release_date": release_date.isoformat(),
        "workdays": sorted(workdays),
        "holidays": sorted(h.isoformat() for h in holidays),
        "artifacts": [a["id"] for a in manifest.get("artifacts", [])],
        "materials": material_records,
        "deduplicated": duplicates,
        "schedule": schedule,
        "items": items,
        "deadlines": {item["id"]: item["deadline"] for item in items},
        "statuses": item_statuses,
        "gaps": {item["id"]: item["gap"] for item in items if item["gap"] is not None},
        "matched_materials": {
            item["id"]: item["matched_materials"]
            for item in items
            if item["matched_materials"]
        },
        "summary": {
            "total": len(items),
            "satisfied": counts["satisfied"],
            "pending": counts["pending"],
            "overdue": counts["overdue"],
        },
    }
    out["snapshot_hash"] = stable_hash(out)
    return out
