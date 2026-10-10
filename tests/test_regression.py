import numpy as np
import pandas as pd
import pytest
from debris_research.regression import error_metrics, fit_regressor


def test_log_error_and_empty_subgroup():
    target = np.array([-6., -8.])
    forecast = np.array([-5., -10.])
    result = error_metrics(target, forecast)
    assert result['mae'] == 1.5
    assert result['rmse'] == pytest.approx(np.sqrt(2.5))
    assert result['mean_signed_error'] == -.5
    assert error_metrics(target, forecast, np.array([False, False]))['mae'] is None


def test_regression_guards_and_saved_predictions(tmp_path):
    import joblib
    train = pd.DataFrame({'partition': ['train']*60, 'current_log_risk': np.tile([-10., -5., np.nan], 20),
                          'target_final_log_risk': np.tile([-9., -4., -8.], 20)})
    model = fit_regressor(train, ['current_log_risk'])
    assert model.named_steps['simpleimputer'].statistics_[0] == -7.5
    inputs = pd.DataFrame({'current_log_risk': [-6., np.nan]})
    predictions = model.predict(inputs)
    assert np.isfinite(predictions).all()
    joblib.dump(model, tmp_path/'model.joblib')
    np.testing.assert_array_equal(predictions, joblib.load(tmp_path/'model.joblib').predict(inputs))
    with pytest.raises(ValueError, match='training only'):
        fit_regressor(train.assign(partition='evaluation'), ['current_log_risk'])
    with pytest.raises(ValueError, match='predeclared'):
        fit_regressor(train, ['target_final_log_risk'])
    with pytest.raises(ValueError, match='finite'):
        fit_regressor(train.assign(target_final_log_risk=np.inf), ['current_log_risk'])
