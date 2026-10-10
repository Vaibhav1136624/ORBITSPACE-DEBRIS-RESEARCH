import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip('torch')
from debris_research.sequence import histories, SequenceScaler, EncounterGRU, FIELDS
from debris_research.data import CONTEXT


def test_sequence_ignores_future_and_preserves_order_and_gaps():
    raw = pd.DataFrame({'event_id': [1]*4, 'time_to_tca': [2., .5, 4., 3.5],
                        'risk': [-6., -1., -10., -9.]})
    for c in CONTEXT:
        raw[c] = 0.
    result = histories(raw, [1])[0]
    np.testing.assert_array_equal(result[:, 0], [-10., -9., -6.])
    np.testing.assert_array_equal(result[:, 2], [0., .5, 1.5])
    raw.loc[raw.time_to_tca < 2., 'risk'] = -30.
    np.testing.assert_array_equal(result, histories(raw, [1])[0])
    short = histories(raw, [1], max_reports=2)[0]
    np.testing.assert_array_equal(short[:, 0], [-9., -6.])
    np.testing.assert_array_equal(short[:, 2], [0., 1.5])


def test_scaler_fits_real_training_tokens_and_marks_missingness():
    train = np.zeros((2, len(FIELDS))); train[:, 0] = [-10., -6.]
    scaler = SequenceScaler().fit([train])
    assert scaler.scaler.mean_[0] == -8.
    val = np.ones((1, len(FIELDS)))*100.
    val[0, 3] = np.nan
    x, lengths = scaler.transform([val])
    assert scaler.scaler.mean_[0] == -8.
    assert lengths.tolist() == [1]
    assert x[0, 0, len(FIELDS)+3] == 1.
    assert torch.count_nonzero(x[:, 1:]) == 0


def test_gru_ignores_padding_and_checkpoint_reloads(tmp_path):
    torch.manual_seed(7)
    model = EncounterGRU().eval()
    x = torch.randn(2, 4, 2*len(FIELDS)); lengths = torch.tensor([1, 3])
    with torch.no_grad():
        before = model(x, lengths)
        x[0, 1:] = 10000.; x[1, 3:] = -10000.
        torch.testing.assert_close(before, model(x, lengths))
        torch.save(model.state_dict(), tmp_path/'weights.pt')
        restored = EncounterGRU().eval()
        restored.load_state_dict(torch.load(tmp_path/'weights.pt', weights_only=True))
        torch.testing.assert_close(before, restored(x, lengths))
