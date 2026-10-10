import numpy as np
import pandas as pd
import pytest
from debris_research.train import fit_model, validate_partitions


def frame(partition, start=0):
    return pd.DataFrame({'event_id': np.arange(start, start+6),
                         'partition': partition, 'target_high_risk': [0, 0, 0, 1, 0, 1],
                         'current_log_risk': [-12., -10., np.nan, -5., -8., -4.]})


def test_rejects_reserved_partitions_and_overlapping_encounters():
    with pytest.raises(ValueError, match='validation'):
        validate_partitions(frame('train'), frame('evaluation', 10))
    with pytest.raises(ValueError, match='overlap'):
        validate_partitions(frame('train'), frame('validation'))


def test_preprocessing_uses_training_only_and_predictions_do_not_refit():
    train = frame('train')
    model = fit_model(train, ['current_log_risk'], 7)
    imputer = model.named_steps['simpleimputer']
    assert imputer.statistics_[0] == -8.
    before = model.named_steps['standardscaler'].mean_.copy()
    validation = frame('validation', 10)
    validation['current_log_risk'] = 1000.
    scores = model.predict_proba(validation[['current_log_risk']])
    np.testing.assert_array_equal(before, model.named_steps['standardscaler'].mean_)
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError, match='training partition'):
        fit_model(validation, ['current_log_risk'], 7)
    with pytest.raises(ValueError, match='predeclared'):
        fit_model(train, ['target_high_risk'], 7)


@pytest.mark.parametrize('kind', ['random_forest', 'gradient_boosting'])
def test_tree_models_handle_missing_inputs_and_serialize(tmp_path, kind):
    import joblib
    train = pd.concat([frame('train', i * 6) for i in range(10)], ignore_index=True)
    model = fit_model(train, ['current_log_risk'], 7, kind)
    validation = frame('validation', 100)
    inputs = validation[['current_log_risk']]
    scores = model.predict_proba(inputs)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=1), 1.)
    joblib.dump(model, tmp_path / 'model.joblib')
    # Parallel tree accumulation can change the last floating-point bit.
    np.testing.assert_allclose(scores, joblib.load(tmp_path / 'model.joblib').predict_proba(inputs),
                               rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError, match='training partition'):
        fit_model(validation, ['current_log_risk'], 7, kind)
    with pytest.raises(ValueError, match='predeclared'):
        fit_model(train, ['target_high_risk'], 7, kind)
