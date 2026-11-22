"""Score transactions with the main model and build what the agent works from.

Usage (from agent/, after modelling/download_data.py):
    python build_cases.py                 # score data/fraudTest.csv with data/fraudTrain.csv as history
    python build_cases.py --limit 20000   # only the first 20,000 test transactions in time order (quick build)

This is the step the status update describes as "the model's output for each flagged transaction (the score, the
features that drove it and the card's recent history)". It scores exactly like modelling/predict.py (same code,
same saved model and threshold), then writes to agent/data/:

  queue.jsonl           one case per flagged transaction, in time order: transaction, score, threshold, the
                        human-readable reasons (reasons.py) and the top contributing features (explain.py)
  transactions.parquet  every transaction (history + input); the input and the validation period carry their score
                        and flag (older history has none): the store the agent's
                        tools read (card history, merchant activity). No fraud labels.
  cards.parquet         cardholder attributes, one row per card (home city/state, age when first seen, job)
  context.json          category fraud rates from the history period (for the HIGH_RISK_CATEGORY reason), the
                        calibration table from the validation period (calibration.py) and build information
  labels.parquet        is_fraud for the scored transactions. Kept apart from everything above, like the stream's
                        labels.jsonl: only evaluate.py reads it, never the agent.

In the AWS pipeline, Lambda would produce the same case record per flagged transaction, and the stores would be
DynamoDB tables (store.py is the only module that reads them).
"""
import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import settings
import config
from data import LABEL
from features import CATEGORICAL_FEATURES, FEATURE_VERSION, NUMERIC_FEATURES, add_history_features
from models import load_model
from predict import load_rows

import calibration
import explain
import reasons

log = logging.getLogger('fraudwatch.agent')
TOP_FEATURES = 8
TXN_COLUMNS = ['trans_num', 'cc_num', 'ts', 'trans_date_trans_time', 'merchant', 'category', 'amt', 'dist_um_km',
               'source']
CARD_COLUMNS = ['cc_num', 'gender', 'age', 'job', 'city', 'state', 'city_pop']


def clean_merchant(name):
    """Every merchant in the Sparkov data is named 'fraud_<name>' (fraud or not); drop the misleading prefix."""
    return name[6:] if isinstance(name, str) and name.startswith('fraud_') else name


def category_context(feats):
    """Fraud rate per category over the labelled history rows (training period): no test labels are used."""
    labelled = feats[(feats.source == 'history') & feats[LABEL].notna()]
    if labelled.empty:
        raise ValueError('the history has no is_fraud labels, so category fraud rates cannot be computed')
    return {'category_fraud_rate': labelled.groupby('category')[LABEL].mean().round(6).to_dict(),
            'overall_fraud_rate': round(float(labelled[LABEL].mean()), 6)}


def validation_calibration(feats, is_val, others, threshold):
    """The calibration table (calibration.py) from the flagged validation rows."""
    val = feats.loc[is_val, ['trans_num', 'cc_num', 'ts', LABEL, 'fraud_probability', 'fraud_flag']]
    flagged = val[val.fraud_flag == 1].copy()
    if flagged.empty or flagged[LABEL].isna().any():
        raise ValueError('the validation period has no labelled flagged transactions to calibrate on')
    flagged['other_flags'] = others.reindex(flagged.trans_num).fillna(0).to_numpy()
    flagged['compromised'] = calibration.card_compromised(flagged, val).reindex(flagged.trans_num).to_numpy()
    return calibration.build_table(flagged.rename(columns={LABEL: 'is_fraud'}), threshold)


def make_cases(flagged, contrib, model, ctx, others):
    """One case record per flagged row (flagged: feature table rows; contrib: per-source-feature contributions;
    others: earlier flagged transactions on the card in the last 72 h, by trans_num)."""
    cases = []
    for idx, row in flagged.iterrows():
        c = contrib.loc[idx]
        found, other = reasons.explain_row(row, c, ctx)
        top = c.drop('bias').sort_values(ascending=False).head(TOP_FEATURES)
        n_others = int(others.get(row.trans_num, 0))
        cases.append({
            'trans_num': row.trans_num,
            'cc_num': int(row.cc_num),
            'card': f'...{str(int(row.cc_num))[-4:]}',
            'time': row.trans_date_trans_time,
            'ts': int(row.ts),
            'merchant': clean_merchant(row.merchant),
            'category': row.category,
            'category_label': reasons.category_label(row.category),
            'amount': round(float(row.amt), 2),
            'fraud_probability': round(float(row.fraud_probability), 4),
            'threshold': round(float(model.threshold), 4),
            'model': model.name,
            'feature_version': model.feature_version,
            'other_flags_72h': n_others,
            'calibration': calibration.lookup(ctx['calibration_table'], float(row.fraud_probability), n_others,
                                              model.threshold),
            'reasons': found,
            'other_weight': other,
            'top_features': [{'feature': f, 'value': (row[f] if f in CATEGORICAL_FEATURES else round(float(row[f]), 4)),
                              'contribution': round(float(v), 3), 'reason': reasons.FEATURE_TO_REASON.get(f, 'OTHER')}
                             for f, v in top.items()],
        })
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, default=config.DATA_DIR / config.TEST_FILE)
    ap.add_argument('--history', type=Path, nargs='*', default=[config.DATA_DIR / config.TRAIN_FILE])
    ap.add_argument('--limit', type=int, help='score only the first N input transactions in time order')
    ap.add_argument('--artifacts-dir', type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument('--out', type=Path, default=settings.DATA_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S', stream=sys.stdout)

    # 1. the main model, exactly as predict.py loads it
    meta = json.loads((args.artifacts_dir / 'metadata.json').read_text())
    model = load_model(args.artifacts_dir / meta['best_model'])
    if (model.numeric, model.categorical, model.feature_version) != (NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                                                                       FEATURE_VERSION):
        sys.exit('the saved model does not match features.py; run modelling/train.py again')
    with open(args.artifacts_dir / 'job_buying_power.pkl', 'rb') as f:
        job = pickle.load(f)
    log.info(f'model: {model.name} | threshold {model.threshold:.4f}')

    # 2. features for history + input (same code as training and predict.py)
    t0 = time.time()
    input_ids, df = load_rows(args.input, args.history, args.limit)
    feats = add_history_features(df)
    del df
    job.transform(feats)
    log.info(f'features built for {len(feats):,} transactions ({time.time() - t0:.0f}s)')

    # 3. score the input rows, and the validation period of the history (the last VAL_FRAC of the training file
    #    by time, as in train.py: not used to fit the model) for the calibration table
    is_input = (feats.source == 'input').to_numpy()
    is_hist = (feats.source == 'history').to_numpy()
    cut = np.quantile(feats.ts[is_hist], 1 - config.VAL_FRAC)
    is_val = is_hist & (feats.ts > cut).to_numpy()
    is_scored = is_input | is_val
    feats['fraud_probability'] = np.nan
    feats.loc[is_scored, 'fraud_probability'] = model.predict_proba(feats[is_scored])
    feats['fraud_flag'] = (feats.fraud_probability >= model.threshold).astype('int8')
    flagged = feats[is_input & (feats.fraud_flag == 1).to_numpy()].sort_values(['ts', 'trans_num'])
    log.info(f'scored {is_input.sum():,} transactions, {len(flagged):,} flagged '
             f'(+ {is_val.sum():,} validation-period transactions for calibration)')

    # 4. cases: reasons, top features and calibration for every flagged transaction
    ctx = category_context(feats)
    others = calibration.prior_flag_counts(feats.loc[is_scored, ['trans_num', 'cc_num', 'ts', 'fraud_flag']])
    ctx['calibration_table'] = validation_calibration(feats, is_val, others, model.threshold)
    log.info('calibration (validation period):\n' + pd.DataFrame(ctx['calibration_table']).to_string(index=False))
    contrib = explain.by_source_feature(explain.contributions(model, flagged), model.encoder)
    # compared as probabilities: near p = 1 the log-odds of p are dominated by floating-point rounding
    gap = np.abs(1 / (1 + np.exp(-contrib.sum(axis=1).to_numpy())) - flagged.fraud_probability.to_numpy())
    if len(gap) and gap.max() > 1e-9:
        raise RuntimeError(f'contributions do not add up to the model score (max gap {gap.max():.2e})')
    cases = make_cases(flagged, contrib, model, ctx, others)

    # 5. write
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / 'queue.jsonl', 'w') as f:
        for case in cases:
            f.write(json.dumps(case) + '\n')
    txns = feats[TXN_COLUMNS + ['fraud_probability', 'fraud_flag']].copy()
    txns['merchant'] = txns.merchant.map(clean_merchant)
    txns.sort_values(['cc_num', 'ts', 'trans_num']).reset_index(drop=True).to_parquet(args.out / 'transactions.parquet')
    cards = feats.sort_values('ts').groupby('cc_num', as_index=False).first()[CARD_COLUMNS + ['ts']]
    cards.rename(columns={'age': 'age_first_seen', 'ts': 'first_seen_ts'}).to_parquet(args.out / 'cards.parquet')
    feats.loc[is_input, ['trans_num', LABEL]].reset_index(drop=True).to_parquet(args.out / 'labels.parquet')
    ctx['build'] = {'model': model.name, 'threshold': model.threshold, 'feature_version': model.feature_version,
                    'input': args.input.name, 'input_rows': int(is_input.sum()), 'flagged': len(cases),
                    'history': [p.name for p in args.history], 'built': time.strftime('%Y-%m-%dT%H:%M:%S')}
    (args.out / 'context.json').write_text(json.dumps(ctx, indent=2))

    n_reasons = pd.Series([len(c['reasons']) for c in cases])
    top = pd.Series([c['reasons'][0]['code'] for c in cases if c['reasons']]).value_counts()
    log.info(f'{len(cases):,} cases -> {config.display_path(args.out / "queue.jsonl")} | reasons per case: '
             f'median {n_reasons.median():.0f}, none for {(n_reasons == 0).sum()} | strongest reason:\n{top.to_string()}')


if __name__ == '__main__':
    main()
