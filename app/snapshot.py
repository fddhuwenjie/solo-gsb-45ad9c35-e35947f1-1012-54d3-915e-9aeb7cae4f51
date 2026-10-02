"""Fulfillment status snapshots for frozen release batches.

A snapshot reads the *immutable* batch manifest (artifacts, merged
components and their obligations -- shared components counted exactly
once), computes the final per-obligation deadline from the submitted
``{offset, unit, depends_on}`` rules, and judges every component
obligation instance against only the materials provided in time:

* ``satisfied`` -- the obligation's nested ``all``/``any`` evidence rule
  already evaluates to true over the in-window materials (a product-level
  material may cover every component; a component-level material covers
  only its declared component; the conjuncts of an ``all`` may be proven
  by different materials, exactly as in fulfillment-check);
* ``overdue``  -- the local ``now`` date is strictly past the deadline and
  the rule is still not satisfied -- a deadline due *on* the local date is
  never overdue;
* ``pending``  -- the deadline has not passed yet and the rule is not
  satisfied.

The deadline window is compared on **timezone dates**: a material counts
toward an instance only when its ``provided_at`` local date is no later
than that obligation's final local deadline, regardless of wall-clock
time.  Materials arriving after the deadline are still reported on the
unsatisfied instance (``gaps[].late_materials``) so callers can tell
"evidence exists but arrived too late" apart from "no evidence at all".

Invalid scheduling or evidence content fails the whole request with
HTTP 422 before any instance is judged -- a snapshot is never produced
partially.  Every output collection is sorted and the ``snapshot_hash``
covers the canonical response, so request ordering (materials, object
keys, workdays/holidays/depends_on) never changes the result.
"""

from datetime import timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .common import ClientError, canonical, stable_hash
from .fulfillment import (
    UnprocessableError,
    eval_circuit,
    _iter_leaf_indexes,
    _leaf_matches,
    _missing_leaves,
    _witness,
    build_formula,
)
from .schedule import (
    MAX_OFFSET,
    UNITS,
    _add_business_days,
    _parse_holiday,
    _parse_release_at,
    _tighten,
)
from .solver import _find_cycles


# ---------------------------------------------------------------------------
# request validation (all content failures surface as HTTP 422)
# ---------------------------------------------------------------------------

def _parse_moment(value, field):
    """Parse an RFC 3339 timestamp for a content field (HTTP 422).

    Scheduling content belongs to the 422 contract of this endpoint, so a
    bad/absent ``now`` or ``provided_at`` value is not a 400 envelope
    error the way a structurally wrong top-level field would be.
    """
    if not isinstance(value, str):
        raise UnprocessableError("%s must be an RFC 3339 timestamp string" % field)
    try:
        return _parse_release_at(value)
    except UnprocessableError:
        raise
    except ClientError:
        raise UnprocessableError(
            "%s %r is not a valid RFC 3339 timestamp" % (field, value)
        )


def _normalize_schedule_part(body):
    """Validate the scheduling fields; every failure is 422.

    Returns ``(now, release_at, tz_name, tz, workdays, holidays, rules)``
    with ``rules`` mapping obligation id to
    ``{"offset", "unit", "depends_on"}`` (deps sorted, deduplicated).
    """
    now = _parse_moment(body.get("now"), "now")
    release_at = _parse_moment(body.get("release_at"), "release_at")

    tz_name = body.get("timezone")
    if not isinstance(tz_name, str) or not tz_name.strip():
        raise UnprocessableError("timezone must be a non-empty IANA timezone name")
    tz_name = tz_name.strip()
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise UnprocessableError("unknown IANA timezone %r" % tz_name)

    raw_workdays = body.get("workdays")
    if not isinstance(raw_workdays, list) or not raw_workdays:
        raise UnprocessableError(
            "workdays must be a non-empty array of ISO weekday numbers "
            "(1=Monday .. 7=Sunday)"
        )
    workdays = set()
    for wd in raw_workdays:
        if isinstance(wd, bool) or not isinstance(wd, int) or not 1 <= wd <= 7:
            raise UnprocessableError(
                "workdays entries must be integers in 1..7 (1=Monday .. 7=Sunday)"
            )
        workdays.add(wd)

    raw_holidays = body.get("holidays", [])
    if not isinstance(raw_holidays, list):
        raise UnprocessableError("holidays must be an array of YYYY-MM-DD date strings")
    holidays = {_parse_holiday(h) for h in raw_holidays}

    raw_rules = body.get("obligations", {})
    if not isinstance(raw_rules, dict) or not raw_rules:
        raise UnprocessableError(
            "obligations must be a non-empty object mapping obligation id "
            "to its deadline rule"
        )
    rules = {}
    for ob in raw_rules:
        if not isinstance(ob, str) or not ob:
            raise UnprocessableError("obligation ids must be non-empty strings")
        spec = raw_rules[ob]
        where = "obligations[%s]" % ob
        if not isinstance(spec, dict):
            raise UnprocessableError("%s must be an object" % where)
        offset = spec.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise UnprocessableError("%s.offset must be a non-negative integer" % where)
        if offset > MAX_OFFSET:
            raise UnprocessableError(
                "%s.offset must not exceed %d" % (where, MAX_OFFSET)
            )
        unit = spec.get("unit")
        if unit not in UNITS:
            raise UnprocessableError("%s.unit must be one of %s" % (where, list(UNITS)))
        raw_deps = spec.get("depends_on", [])
        if not isinstance(raw_deps, list):
            raise UnprocessableError(
                "%s.depends_on must be an array of obligation ids" % where
            )
        deps = set()
        for dep in raw_deps:
            if not isinstance(dep, str) or not dep:
                raise UnprocessableError(
                    "%s.depends_on entries must be non-empty obligation ids" % where
                )
            deps.add(dep)
        rules[ob] = {"offset": offset, "unit": unit, "depends_on": sorted(deps)}
    return now, release_at, tz_name, tz, workdays, holidays, rules


def _normalize_materials(body, valid_components):
    """Validate the materials array (422 on every content problem).

    Returns a sorted list of
    ``{"id", "digest", "type", "scope", "component", "provided_at",
       "provided_date"}``.  Materials with the same id are rejected; unlike
    fulfillment-check, identical digests are **not** merged -- each
    submission is a distinct piece of evidence with its own ``provided_at``.
    """
    raw_list = body.get("materials", [])
    if not isinstance(raw_list, list):
        raise UnprocessableError("materials must be an array")

    materials = []
    seen = set()
    for i, raw in enumerate(raw_list):
        where = "materials[%d]" % i
        if not isinstance(raw, dict):
            raise UnprocessableError("%s must be an object" % where)
        mid = raw.get("id")
        if not isinstance(mid, str) or not mid.strip():
            raise UnprocessableError("%s.id must be a non-empty string" % where)
        mid = mid.strip()
        mtype = raw.get("type")
        if not isinstance(mtype, str) or not mtype.strip():
            raise UnprocessableError("%s.type must be a non-empty string" % where)
        mtype = mtype.strip()
        scope = raw.get("scope")
        if scope not in ("product", "component"):
            raise UnprocessableError(
                "%s.scope must be one of ['product', 'component']" % where
            )
        component = None
        if scope == "component":
            component = raw.get("component")
            if not isinstance(component, str) or not component.strip():
                raise UnprocessableError(
                    "%s: component-scoped materials must declare a non-empty "
                    "component" % where
                )
            component = component.strip()
            if component not in valid_components:
                raise UnprocessableError(
                    "%s.component references component %r that is not part of "
                    "the release batch" % (where, component)
                )
        elif raw.get("component") is not None:
            raise UnprocessableError(
                "%s: product-scoped materials must not declare a component" % where
            )
        digest = raw.get("digest")
        if digest is not None and (not isinstance(digest, str) or not digest.strip()):
            raise UnprocessableError(
                "%s.digest must be a non-empty string when present" % where
            )
        if isinstance(digest, str):
            digest = digest.strip()
        provided = _parse_moment(raw.get("provided_at"), "%s.provided_at" % where)
        if mid in seen:
            raise UnprocessableError("duplicate material id %r" % mid)
        seen.add(mid)
        materials.append(
            {
                "id": mid,
                "digest": digest,
                "type": mtype,
                "scope": scope,
                "component": component,
                "provided_at": provided,
            }
        )
    return sorted(materials, key=lambda m: m["id"])


def _normalize_evidence(body, valid_components):
    """Validate the nested all/any evidence rules (422 on any problem).

    Mirrors :func:`fulfillment.normalize_rules` but every structural or
    semantic failure -- including a rule leaf pinned to a component that
    is not part of the batch -- is an :class:`UnprocessableError`.
    """
    raw = body.get("rules")
    if raw is None:
        return {"obligations": {}}
    if not isinstance(raw, dict):
        raise UnprocessableError("rules must be an object")
    raw_obs = raw.get("obligations")
    if not isinstance(raw_obs, dict):
        raise UnprocessableError("rules.obligations must be an object")
    valid_components = set(valid_components)

    def norm_node(node, path):
        if not isinstance(node, dict):
            raise UnprocessableError("%s must be an object" % path)
        keys = set(node)
        combinators = keys & {"all", "any"}
        if len(combinators) > 1:
            raise UnprocessableError("%s: 'all' and 'any' are mutually exclusive" % path)
        if combinators:
            comb = next(iter(combinators))
            children = node[comb]
            if not isinstance(children, list) or not children:
                raise UnprocessableError("%s.%s must be a non-empty array" % (path, comb))
            extra = keys - {comb}
            if extra:
                raise UnprocessableError(
                    "%s: %s nodes may not carry leaf fields %s"
                    % (path, comb, sorted(extra))
                )
            return {
                comb: [
                    norm_node(ch, "%s.%s[%d]" % (path, comb, i))
                    for i, ch in enumerate(children)
                ]
            }
        unknown = keys - {"type", "scope", "component"}
        if unknown:
            raise UnprocessableError("%s: unknown fields %s" % (path, sorted(unknown)))
        leaf = {}
        if "type" in node:
            if not isinstance(node["type"], str) or not node["type"].strip():
                raise UnprocessableError("%s.type must be a non-empty string" % path)
            leaf["type"] = node["type"].strip()
        if "scope" in node:
            if node["scope"] not in ("product", "component"):
                raise UnprocessableError(
                    "%s.scope must be one of ['product', 'component']" % path
                )
            leaf["scope"] = node["scope"]
        if "component" in node:
            if not isinstance(node["component"], str) or not node["component"].strip():
                raise UnprocessableError(
                    "%s.component must be a non-empty string" % path
                )
            cid = node["component"].strip()
            if cid not in valid_components:
                raise UnprocessableError(
                    "%s.component references unknown component %r" % (path, cid)
                )
            leaf["component"] = cid
        if not leaf:
            raise UnprocessableError(
                "%s: a leaf rule must constrain at least one field" % path
            )
        return leaf

    rules = {}
    for ob in raw_obs:
        if not isinstance(ob, str) or not ob:
            raise UnprocessableError("obligation rule keys must be non-empty strings")
        rules[ob] = norm_node(raw_obs[ob], "rules.obligations[%s]" % ob)
    return {"obligations": rules}


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def _utc_z(dt):
    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_batch_snapshot(body, manifest, batch_id):
    """Build the fulfillment status snapshot for a frozen batch manifest.

    The API layer guarantees the batch exists (404 otherwise) and that
    *body* is a JSON object (400 otherwise); every scheduling/evidence
    content failure raises :class:`UnprocessableError` (422) before an
    instance is judged, so a partial snapshot can never be returned.
    """
    artifact_obligations = sorted(manifest.get("artifact_obligations", []))
    components = sorted(
        (c["id"] for c in manifest.get("components", [])),
    )

    # The path identifies the batch; a body batch_id, when present, must
    # agree with it -- otherwise the request names two different batches.
    body_batch = body.get("batch_id")
    if body_batch is not None and (
        not isinstance(body_batch, str) or body_batch.strip() != batch_id
    ):
        raise UnprocessableError(
            "body batch_id %r does not match path batch_id %r"
            % (body_batch, batch_id)
        )

    (
        now,
        release_at,
        tz_name,
        tz,
        workdays,
        holidays,
        rules,
    ) = _normalize_schedule_part(body)
    materials = _normalize_materials(body, set(components))
    evidence = _normalize_evidence(body, components)

    # Every artifact obligation must have a deadline rule; the merged
    # component obligations are a subset of the artifact union, so each
    # instance then has a well-defined deadline.
    missing = [o for o in artifact_obligations if o not in rules]
    if missing:
        raise UnprocessableError(
            "missing scheduling rules for batch artifact obligations: %s"
            % ", ".join(missing)
        )
    dangling = sorted(
        {d for r in rules.values() for d in r["depends_on"] if d not in rules}
    )
    if dangling:
        raise UnprocessableError(
            "depends_on references obligations without rules: %s"
            % ", ".join(dangling)
        )
    edges = [(d, ob) for ob, r in rules.items() for d in r["depends_on"]]
    cycles = _find_cycles(sorted(rules), edges)
    if cycles:
        raise UnprocessableError(
            "obligation dependency cycle detected: %s"
            % "; ".join(" -> ".join(c + [c[0]]) for c in cycles)
        )

    # component coverage per obligation, taken from the frozen manifest
    components_by_ob = {o: [] for o in rules}
    for crep in manifest.get("components", []):
        for ob in crep.get("obligations", []):
            components_by_ob.setdefault(ob, []).append(crep["id"])
    for ob in components_by_ob:
        components_by_ob[ob].sort()

    release_date = release_at.astimezone(tz).date()
    now_date = now.astimezone(tz).date()

    raw = {}
    for ob, r in rules.items():
        if r["unit"] == "calendar_day":
            raw[ob] = release_date + timedelta(days=r["offset"])
        else:
            raw[ob] = _add_business_days(
                release_date, r["offset"], workdays, holidays
            )
    final, chains = _tighten(rules, raw)

    # localize each material once: all window comparisons are timezone dates
    prepared = []
    for m in materials:
        m2 = dict(m)
        m2["provided_date"] = m["provided_at"].astimezone(tz).date()
        prepared.append(m2)
    materials_by_id = {m["id"]: m for m in prepared}
    rule_by_ob = evidence["obligations"]

    # one instance per (merged component, obligation): shared components
    # were already deduplicated by the frozen manifest, so each counts once
    instances = []
    for crep in sorted(manifest.get("components", []), key=lambda c: c["id"]):
        for ob in sorted(crep.get("obligations", [])):
            instances.append({"component": crep["id"], "obligation": ob})

    items = []
    counts = {"satisfied": 0, "overdue": 0, "pending": 0}
    for inst in instances:
        ob = inst["obligation"]
        cid = inst["component"]
        deadline = final[ob]

        # in-window materials for this instance: scope/coverage matches and
        # the local provided date is no later than the local deadline date
        in_window = []
        late = []
        for m in prepared:
            if m["scope"] == "component" and m["component"] != cid:
                continue
            if m["provided_date"] <= deadline:
                in_window.append(m)
            else:
                late.append(m)
        in_ids = [m["id"] for m in in_window]

        rule_node = rule_by_ob.get(ob)
        satisfied = False
        witness_ids = []
        gap = None
        if rule_node is not None:
            formula = build_formula(materials_by_id, in_ids, rule_node, inst)
            tree, leaf_masks, leaves, gates, root, nleaves = formula
            truth = 0
            for mask in leaf_masks:
                truth |= mask

            # per-leaf ids of in-window materials matching that leaf
            leaf_mats = tuple(
                tuple(
                    in_ids[j]
                    for j in range(len(in_ids))
                    if (leaf_masks[j] >> li) & 1
                )
                for li in range(len(leaves))
            )
            if eval_circuit(gates, root, truth, nleaves):
                satisfied = True
                witness_ids = list(_witness(tree, leaf_mats, set(in_ids)) or ())
            else:
                # gap analysis runs against in-window evidence only; a
                # material arriving after the deadline can never close it
                any_leaf_matched = any(
                    leaf_mats[li] for li in _iter_leaf_indexes(tree)
                )
                missing_idx = _missing_leaves(tree, leaf_mats)
                candidate = sorted(
                    {
                        mid
                        for li in _iter_leaf_indexes(tree)
                        for mid in leaf_mats[li]
                    }
                )
                missing_leaves_out = sorted(
                    (leaves[li] for li in missing_idx),
                    key=canonical,
                )
                late_ids = sorted(
                    m["id"]
                    for m in late
                    if any(_leaf_matches(leaf, m, inst) for leaf in leaves)
                )
                gap = {
                    "reason": (
                        "no-matching-material"
                        if not any_leaf_matched
                        else "incomplete-evidence"
                    ),
                    "missing_leaves": (
                        [] if not any_leaf_matched else missing_leaves_out
                    ),
                    "candidate_materials": candidate,
                    "late_materials": late_ids,
                }
        else:
            # No evidence rule was submitted for this obligation: nothing
            # can be judged as evidence, so it can never be satisfied.
            gap = {
                "reason": "no-evidence-rule",
                "missing_leaves": [],
                "candidate_materials": [],
                "late_materials": [],
            }

        if satisfied:
            status = "satisfied"
        elif now_date > deadline:
            status = "overdue"
        else:
            status = "pending"
        counts[status] += 1

        item = {
            "component": cid,
            "obligation": ob,
            "deadline": deadline.isoformat(),
            "status": status,
            "satisfied_by": witness_ids,
            "gaps": [gap] if gap is not None else [],
        }
        items.append(item)

    schedule = []
    for ob in sorted(rules):
        r = rules[ob]
        tightened = final[ob] != raw[ob]
        schedule.append(
            {
                "obligation": ob,
                "components": components_by_ob.get(ob, []),
                "offset": r["offset"],
                "unit": r["unit"],
                "depends_on": r["depends_on"],
                "raw_deadline": raw[ob].isoformat(),
                "deadline": final[ob].isoformat(),
                "tightened": tightened,
                "tightening_chain": chains[ob] if tightened else [],
            }
        )

    out_materials = [
        {
            "id": m["id"],
            "digest": m["digest"],
            "type": m["type"],
            "scope": m["scope"],
            "component": m["component"],
            "provided_at": _utc_z(m["provided_at"]),
            "provided_date": m["provided_date"].isoformat(),
        }
        for m in prepared
    ]

    out = {
        "batch_id": batch_id,
        "batch_hash": manifest["batch_hash"],
        "now": _utc_z(now),
        "release_at": _utc_z(release_at),
        "timezone": tz_name,
        "release_date": release_date.isoformat(),
        "now_date": now_date.isoformat(),
        "workdays": sorted(workdays),
        "holidays": sorted(h.isoformat() for h in holidays),
        "materials": out_materials,
        "schedule": schedule,
        "instances": items,
        "summary": {
            "total": len(items),
            "satisfied": counts["satisfied"],
            "overdue": counts["overdue"],
            "pending": counts["pending"],
        },
    }
    out["snapshot_hash"] = stable_hash(out)
    return out
