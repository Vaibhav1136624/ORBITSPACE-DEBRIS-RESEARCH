"""Inspectable research indicators; no trained probability or manoeuvre claims."""
from datetime import datetime
import math
from statistics import mean
from .messages import utc, digest


FEATURES = [
    ('value_of_information', 'experimental_proxy', 'Review urgency from uncertainty and time remaining; not causal observation value.'),
    ('horizon_forecast', 'experimental_proxy', 'Same-provider log-risk trend; requires prospective horizon calibration.'),
    ('consequence_ranking', 'conditional_proxy', 'Probability-weighted relative kinetic energy when both masses and speed are available.'),
    ('source_disagreement', 'conditional', 'Explicitly associated, contemporaneous provider reports only.'),
    ('capacity_abstention', 'rules_only', 'Missing, stale and conflicting inputs defer to review; no calibration claim.'),
    ('counterfactual', 'implemented_policy_explanation', 'Risk change needed for an assigned slot under the fixed queue policy.'),
    ('notification_suppression', 'experimental_policy', 'Durable transitions, worsening risk and deadline reminders; local records only.'),
    ('orbital_graph', 'conditional_proxy', 'Encounter connectivity with real object identifiers; not a cascade simulator.'),
    ('provider_profiles', 'implemented_descriptive', 'Observed latency and completeness are not accuracy scores.'),
    ('human_disagreement', 'collection_only', 'Versioned analyst feedback; learning awaits adjudicated labels.'),
]


def seconds(later, earlier):
    return (datetime.fromisoformat(later)-datetime.fromisoformat(earlier)).total_seconds()


def evaluate(reports, as_of, capacity=5, threshold=-6, stale_hours=24, disagreement_hours=6):
    """Only messages created AND received by as_of can influence decisions."""
    as_of = utc(as_of)
    if type(capacity) is not int or capacity < 1 or capacity > 1000:
        raise ValueError('Capacity must be between 1 and 1000')
    groups = {}
    for report in reports:
        if report['created_at'] <= as_of and report['received_at'] <= as_of and report['tca'] > as_of:
            # Unassociated messages cannot be used as cross-provider corroboration.
            key = report['encounter_key']
            if report['association'] != 'explicit': key = report['provider']+':'+key
            groups.setdefault(key, []).append(report)
    rows = []
    for key, history in groups.items():
        pairs = {tuple(sorted(r['object_ids'])) for r in history}
        history.sort(key=lambda r: (r['created_at'], r['received_at'], r['message_id']))
        latest = history[-1]
        providers = {}
        for r in history: providers[r['provider']] = r
        recent = [r for r in providers.values() if seconds(as_of, r['created_at']) <= disagreement_hours*3600]
        risks = [r['reported_log_risk'] for r in recent if r['reported_log_risk'] is not None]
        disagreement = max(risks)-min(risks) if len(risks) > 1 else None
        # Preserve any latest-provider hard escalation even when another feed reports lower risk.
        escalation = any(r['reported_log_risk'] is not None and r['reported_log_risk'] >= threshold for r in providers.values())
        age = seconds(as_of, latest['created_at'])/3600
        hours = min(seconds(r['tca'], as_of)/3600 for r in providers.values())
        reasons = []
        if latest['collision_probability'] is None: reasons.append('missing_probability')
        if latest['zero_pc_reported']: reasons.append('zero_probability_requires_review')
        if any(seconds(as_of,r['created_at']) > stale_hours*3600 for r in providers.values()): reasons.append('stale_provider_report')
        if disagreement is not None and disagreement >= 1: reasons.append('provider_disagreement')
        if len(pairs) != 1: reasons.append('association_object_mismatch')
        if latest['association'] != 'explicit': reasons.append('encounter_association_unverified')
        risk = latest['reported_log_risk']
        same = [r for r in history if r['provider'] == latest['provider'] and r['reported_log_risk'] is not None]
        slope = None
        if len(same) >= 2:
            elapsed = seconds(same[-1]['created_at'], same[-2]['created_at'])/3600
            if elapsed > 0: slope = (same[-1]['reported_log_risk']-same[-2]['reported_log_risk'])/elapsed
        forecast = min(0, risk+slope*min(24, hours)) if slope is not None and risk is not None else None
        masses, speed, pc = latest['masses_kg'], latest['relative_speed_m_s'], latest['collision_probability']
        energy = None
        if all(m is not None and m > 0 for m in masses) and speed is not None and pc is not None:
            reduced_mass = masses[0]*masses[1]/sum(masses)
            candidate = pc*0.5*reduced_mass*speed**2
            energy = candidate if math.isfinite(candidate) else None
        tier = 0 if escalation else 1 if reasons else 2
        rows.append({'encounter_key':key, 'object_ids':latest['object_ids'], 'fixture':any(r['fixture'] for r in history),
                     'latest_message_id':latest['message_id'], 'provider_count':len(providers),
                     'log_risk':risk, 'hours_to_tca':hours, 'report_age_hours':age,
                     'escalation':escalation, 'abstain':bool(reasons), 'review_reasons':reasons,
                     'provider_spread_log10':disagreement, 'forecast_log_risk_proxy':forecast,
                     'forecast_horizon_hours':min(24,hours), 'trend_log10_per_hour':slope,
                     'expected_relative_energy_j_proxy':energy,
                     'review_urgency_proxy':(len(reasons)+(disagreement or 0))/(1+hours),
                     'tier':tier})
    def order(r): return (r['tier'], -(r['log_risk'] if r['log_risk'] is not None else -math.inf), r['encounter_key'])
    rows.sort(key=order)
    for i, row in enumerate(rows):
        row.update(rank=i+1, selected=i < capacity, mandatory_overflow=i >= capacity and row['tier'] < 2)
        if i < capacity:
            counter = {'status':'already_selected'}
        else:
            cutoff = rows[capacity-1]
            if row['tier'] != cutoff['tier'] or row['log_risk'] is None or cutoff['log_risk'] is None or cutoff['log_risk'] >= 0:
                counter = {'status':'requires_policy_tier_or_data_change'}
            else:
                target = math.nextafter(cutoff['log_risk'], math.inf)
                counter = {'status':'risk_only_fixed_policy', 'target_log_risk':target,
                           'increase_log10':max(0,target-row['log_risk']), 'meaning':'Queue sensitivity, not a physical intervention.'}
        row['counterfactual'] = counter
    graph = {}
    for row in rows:
        for object_id in row['object_ids']:
            node = graph.setdefault(object_id, {'object_id':object_id,'encounter_degree':0,'fixture':False})
            node['encounter_degree'] += 1; node['fixture'] |= row['fixture']
    policy = {'version':'utc-research-v1','capacity':capacity,'escalation_log_risk':threshold,
              'stale_hours':stale_hours,'disagreement_hours':disagreement_hours}
    result = {'as_of':as_of, 'policy':policy,'rows':rows,
              'mandatory_overflow':sum(r['mandatory_overflow'] for r in rows),
              'graph':{'nodes':list(graph.values()),'edges':[{'objects':r['object_ids'],'encounter_key':r['encounter_key']} for r in rows]},
              'features':[{'name':n,'status':s,'description':d} for n,s,d in FEATURES],
              'operationally_validated':False}
    result['decision_sha256'] = digest(result)
    return result


def provider_statistics(reports):
    result = []
    for provider in sorted({r['provider'] for r in reports}):
        subset = [r for r in reports if r['provider'] == provider]
        result.append({'provider':provider,'reports':len(subset),
                       'mean_receipt_latency_seconds':mean(seconds(r['received_at'],r['created_at']) for r in subset),
                       'probability_completeness':mean(r['collision_probability'] is not None for r in subset),
                       'explicit_association_fraction':mean(r['association'] == 'explicit' for r in subset),
                       'accuracy':'unknown_without_adjudicated_truth'})
    return result
