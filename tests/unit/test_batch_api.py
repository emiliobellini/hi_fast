"""Unit tests for the public batch-first HiFast API."""
import json

import numpy as np
import pytest

from hi_fast import HiFast
import hi_fast.io as io_module
import hi_fast.spectra as spectra_module


class FakeParams:
    """Minimal replacement for Params that records no external state."""

    _ranges_by_region = {
        'thin': {'a': [0.0, 10.0], 'b': [0.0, 10.0],
                 'z_pk': [0.0, 1.0]},
        'std': {'a': [-10.0, 20.0], 'b': [-10.0, 20.0],
                'z_pk': [0.0, 2.0]},
        'ext': {'a': [-100.0, 100.0], 'b': [-100.0, 100.0],
                'z_pk': [0.0, 10.0]},
    }

    def get(self, params, **kwargs):
        return params.copy()

    def _check_input_param_names(self, params):
        return

    def _check_output_param_values(self, params, trusted_region='ext'):
        ranges = self._ranges_by_region[trusted_region]
        for name, value in params.items():
            if name not in ranges:
                continue
            low, high = ranges[name]
            if not low <= value <= high:
                raise ValueError(
                    '{} outside the {} trusted region'.format(
                        name, trusted_region))

    def is_in_bounds(self, params, trusted_region='ext'):
        try:
            self._check_output_param_values(params, trusted_region)
        except ValueError:
            return False
        return True


class FakePk:
    name = 'pk_m'
    input_params_names = ['a', 'b']
    ref = {'params': {'ln_A_s_1e10': 3.0, 'n_s': 0.96}}
    k_min = 0.0
    k_max = 10.0

    def __init__(self):
        self.batch_lengths = []
        self.class_calls = []

    def _values(self, k, z, params, offset):
        self.batch_lengths.append(len(params))
        k = np.asarray(k)
        return np.asarray([
            row['a'] + 2*row['b'] + z_one + offset + k
            for row, z_one in zip(params, z)
        ])

    def get(self, k, z, params):
        return self._values(k, z, params, offset=0.0)

    def get_fk(self, k, z, params):
        return self._values(k, z, params, offset=10.0)

    def _check_k_values(self, k):
        if np.min(k) < self.k_min or np.max(k) > self.k_max:
            raise ValueError('k outside emulator support')

    def get_from_class(self, k, z, params, **kwargs):
        """Return the native CLASS orientation: (n_k, n_z)."""
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        self.class_calls.append((params.copy(), z.copy(), kwargs))
        return 100.0 + k[:, np.newaxis] + z[np.newaxis, :]

    def get_fk_from_class(self, k, z, params, **kwargs):
        return self.get_from_class(k, z, params, **kwargs)


class FakeFk(FakePk):
    name = 'fk_m'
    input_params_names = ['a', 'b']

    def get(self, k, z, params):
        return self._values(k, z, params, offset=20.0)


class FakeCell:
    name = 'cl_TT_lensed'
    input_params_names = ['a', 'b']
    ell_min = 2
    ell_max = 3000

    def __init__(self):
        self.batch_lengths = []
        self.class_calls = []

    def get(self, ell, params):
        self.batch_lengths.append(len(params))
        ell = np.asarray(ell)
        return np.asarray([
            row['a'] + 2*row['b'] + ell for row in params
        ])

    def _check_ell_values(self, ell):
        if np.min(ell) < self.ell_min or np.max(ell) > self.ell_max:
            raise ValueError('ell outside emulator support')

    def get_from_class(self, ell, params, **kwargs):
        """Return the native one-cosmology CLASS orientation."""
        self.class_calls.append((params.copy(), kwargs))
        return 100.0 + np.atleast_1d(ell).astype(float)


@pytest.fixture
def hifast():
    """Construct HiFast without loading TensorFlow models from disk."""
    instance = HiFast.__new__(HiFast)
    pk = FakePk()
    fk = FakeFk()
    cell = FakeCell()
    instance._spectra = {
        pk.name: pk,
        fk.name: fk,
        cell.name: cell,
    }
    instance._params = {
        name: FakeParams() for name in instance._spectra
    }
    return instance


def test_get_params_names_returns_public_array_order(hifast):
    assert hifast.get_params_names('pk_m') == ['a', 'b']
    assert hifast.get_params_names('fk_m') == ['a', 'b']
    assert hifast.get_params_names('cl_TT_lensed') == ['a', 'b']


def test_get_params_names_uses_pk_fallback_for_missing_fk(hifast):
    del hifast._spectra['fk_m']
    hifast._spectra['pk_m'].input_params_names = [
        'a', 'b', 'ln_A_s_1e10', 'n_s']
    assert hifast.get_params_names('fk_m') == ['a', 'b']


@pytest.mark.parametrize('method, kwargs', [
    ('get_pk', {'name': 'm'}),
    ('get_fk', {'name': 'm', 'get_from_pk': False}),
    ('get_fk', {'name': 'm', 'get_from_pk': True}),
])
def test_dictionary_and_array_inputs_are_equivalent(hifast, method, kwargs):
    dictionaries = [{'a': 1.0, 'b': 2.0}, {'a': 3.0, 'b': 4.0}]
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    z = np.array([0.0, 1.0])
    k = np.array([0.1, 0.2])

    from_dicts = getattr(hifast, method)(k, z, dictionaries, **kwargs)
    from_array = getattr(hifast, method)(k, z, array, **kwargs)
    np.testing.assert_allclose(from_array, from_dicts)


def test_cell_dictionary_and_array_inputs_are_equivalent(hifast):
    dictionaries = [{'a': 1.0, 'b': 2.0}, {'a': 3.0, 'b': 4.0}]
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    ell = np.array([2, 10])
    expected = hifast.get_cell(ell, dictionaries, name='TT')
    result = hifast.get_cell(ell, array, name='TT')
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('method, args, kwargs', [
    ('get_pk', ([0.1], [0.5], {'a': 1.0, 'b': 2.0}), {'name': 'm'}),
    ('get_fk', ([0.1], [0.5], {'a': 1.0, 'b': 2.0}), {'name': 'm'}),
    ('get_cell', ([2], {'a': 1.0, 'b': 2.0}), {'name': 'TT'}),
])
def test_getters_reject_unknown_boundary_options(
        hifast, method, args, kwargs):
    function = getattr(hifast, method)
    with pytest.raises(ValueError, match='trusted_region'):
        function(*args, trusted_region='wide', **kwargs)
    with pytest.raises(ValueError, match='on_out_of_bounds'):
        function(*args, on_out_of_bounds='warn', **kwargs)


def test_pk_mixed_redshifts_use_emulator_and_class(hifast):
    k = np.array([0.1, 0.2])
    z = np.array([0.5, 1.5])
    params = {'a': 1.0, 'b': 2.0}

    result = hifast.get_pk(
        k, z, params, name='m', trusted_region='thin',
        on_out_of_bounds='class', class_precision=2)

    expected = np.array([
        [1.0 + 2*2.0 + 0.5 + k],
        [100.0 + 1.5 + k],
    ]).reshape(1, 2, 2)
    np.testing.assert_allclose(result, expected)
    assert len(hifast._spectra['pk_m'].class_calls) == 1
    assert hifast._spectra['pk_m'].class_calls[0][2]['precision'] == 2


def test_pk_mixed_cosmologies_use_class_only_outside_region(hifast):
    k = np.array([0.1, 0.2])
    params = [
        {'a': 1.0, 'b': 2.0},
        {'a': 20.0, 'b': 2.0},
    ]

    result = hifast.get_pk(
        k, 0.5, params, name='m', trusted_region='thin',
        on_out_of_bounds='class')

    np.testing.assert_allclose(
        result[0, 0], 1.0 + 2*2.0 + 0.5 + k)
    np.testing.assert_allclose(result[1, 0], 100.0 + 0.5 + k)
    assert len(hifast._spectra['pk_m'].class_calls) == 1


@pytest.mark.parametrize('method, kwargs', [
    ('get_pk', {'name': 'm'}),
    ('get_fk', {'name': 'm', 'get_from_pk': False}),
    ('get_fk', {'name': 'm', 'get_from_pk': True}),
])
def test_none_trusted_region_always_uses_class(
        hifast, method, kwargs):
    result = getattr(hifast, method)(
        [0.1, 0.2], [0.5, 1.5], {'a': 20.0, 'b': 2.0},
        trusted_region=None, class_precision=1, **kwargs)

    expected = np.array([[
        100.0 + 0.5 + np.array([0.1, 0.2]),
        100.0 + 1.5 + np.array([0.1, 0.2]),
    ]])
    np.testing.assert_allclose(result, expected)


def test_fk_class_fallback_works_without_dedicated_fk(hifast):
    del hifast._spectra['fk_m']
    del hifast._params['fk_m']

    result = hifast.get_fk(
        [0.1, 0.2], 0.5, {'a': 1.0, 'b': 2.0}, name='m',
        trusted_region=None)

    np.testing.assert_allclose(
        result, [[[100.6, 100.7]]])


def test_outside_region_raises_by_default(hifast):
    with pytest.raises(ValueError, match='thin trusted region'):
        hifast.get_pk(
            [0.1], 0.5, {'a': 20.0, 'b': 2.0}, name='m',
            trusted_region='thin')


def test_disabling_parameter_value_checks_keeps_axis_checks(hifast):
    result = hifast.get_pk(
        [0.1], 0.5, {'a': 20.0, 'b': 2.0}, name='m',
        trusted_region='thin', check_params_values=False)
    np.testing.assert_allclose(result, [[[24.6]]])

    with pytest.raises(ValueError, match='z_pk'):
        hifast.get_pk(
            [0.1], 1.5, {'a': 20.0, 'b': 2.0}, name='m',
            trusted_region='thin', check_params_values=False)


def test_cell_can_fall_back_per_cosmology(hifast):
    ell = np.array([2, 10])
    params = [
        {'a': 1.0, 'b': 2.0},
        {'a': 20.0, 'b': 2.0},
    ]

    result = hifast.get_cell(
        ell, params, name='TT', trusted_region='thin',
        on_out_of_bounds='class')

    np.testing.assert_allclose(result[0], 1.0 + 2*2.0 + ell)
    np.testing.assert_allclose(result[1], 100.0 + ell)
    assert len(hifast._spectra['cl_TT_lensed'].class_calls) == 1


def test_spectral_axis_outside_support_uses_class(hifast):
    pk = hifast.get_pk(
        [20.0], 0.5, {'a': 1.0, 'b': 2.0}, name='m',
        on_out_of_bounds='class')
    cell = hifast.get_cell(
        [4000], {'a': 1.0, 'b': 2.0}, name='TT',
        on_out_of_bounds='class')

    np.testing.assert_allclose(pk, [[[120.5]]])
    np.testing.assert_allclose(cell, [[4100.0]])


def test_cell_getters_fall_back_to_raw_emulator(hifast):
    cell = hifast._spectra.pop('cl_TT_lensed')
    params_handler = hifast._params.pop('cl_TT_lensed')
    cell.name = 'cl_TT'
    hifast._spectra[cell.name] = cell
    hifast._params[cell.name] = params_handler

    params = {'a': 1.0, 'b': 2.0}
    emulator = hifast.get_cell([2, 10], params, name='TT')
    from_class = hifast.get_cell_from_class([2, 10], params, name='TT')

    assert emulator.shape == (1, 2)
    assert from_class.shape == (1, 2)


def test_unknown_cell_name_is_informative(hifast):
    with pytest.raises(ValueError, match="CMB spectrum 'XX'"):
        hifast.get_cell([2], {'a': 1.0, 'b': 2.0}, name='XX')


def test_print_info_can_show_stored_trust_regions(capsys):
    class SimpleSpectrum:
        name = 'pk_m'
        k_min = 0.001
        k_max = 50.0
        z_min = 0.0
        z_max = 10.0
        ell_min = None
        ell_max = None

    class SimpleParams:
        _required = ['h']
        _ranges = {'z_pk': [0.0, 10.0], 'h': [0.5, 0.9]}
        _ranges_by_region = {
            'thin': {'z_pk': [0.0, 2.0], 'h': [0.65, 0.73]},
            'std': {'z_pk': [0.0, 3.0], 'h': [0.6, 0.8]},
            'ext': {'z_pk': [0.0, 10.0], 'h': [0.5, 0.9]},
        }
        _derived = {'h': ['H0', 'h']}

    spectra = {'pk_m': SimpleSpectrum()}
    params = {'pk_m': SimpleParams()}

    io_module._print_info(spectra, params)
    output = capsys.readouterr().out
    assert 'HiFast emulator summary' in output
    assert 'Power spectra' in output
    assert 'get_pk(..., name="m")' in output
    assert 'h' in output
    assert '[0, 3]' in output

    io_module._print_info(spectra, params, name='pk_m')
    output = capsys.readouterr().out
    assert 'HiFast emulator info' in output
    assert 'Thin' in output
    assert 'Std' in output
    assert 'Ext' in output
    assert '[0.65, 0.73]' in output

    io_module._print_info(spectra, params, name='pk_m', bounds='std')
    output = capsys.readouterr().out
    assert 'Min' in output
    assert 'Max' in output
    assert '0.6' in output
    assert '0.8' in output


def test_print_info_can_render_markdown(capsys, tmp_path):
    class SimpleSpectrum:
        name = 'pk_m'
        k_min = 0.001
        k_max = 50.0
        z_min = 0.0
        z_max = 10.0
        ell_min = None
        ell_max = None

    class SimpleParams:
        _required = ['h']
        _ranges = {'z_pk': [0.0, 10.0], 'h': [0.5, 0.9]}
        _ranges_by_region = {
            'thin': {'z_pk': [0.0, 2.0], 'h': [0.65, 0.73]},
            'std': {'z_pk': [0.0, 3.0], 'h': [0.6, 0.8]},
            'ext': {'z_pk': [0.0, 10.0], 'h': [0.5, 0.9]},
        }
        _derived = {'h': ['H0', 'h']}

    spectra = {'pk_m': SimpleSpectrum()}
    params = {'pk_m': SimpleParams()}

    content = io_module._print_info(spectra, params, markdown=True)
    output = capsys.readouterr().out
    assert content == output
    assert '# HiFast Emulator Summary' in content
    assert '| Observable | Public call | Required inputs |' in content
    assert '## Detailed Trust Regions' in content
    assert '| h | [0.65, 0.73] | [0.6, 0.8] | [0.5, 0.9] | `H0` |' in content

    output_path = tmp_path / 'README.md'
    written = io_module._print_info(
        spectra, params, name='pk_m', bounds='std',
        markdown=True, output=str(output_path))
    assert output_path.read_text() == written
    assert '# HiFast Emulator: pk_m' in written
    assert '| h | 0.6 | 0.8 | `H0` |' in written


def test_print_info_shows_background_nomenclature(capsys):
    background = [{
        'name': 'H',
        'hiclassy': 'Hubble(z)',
        'input': 'z',
        'units': 'km/s/Mpc',
    }]

    io_module._print_info({}, {}, background=background)
    terminal = capsys.readouterr().out
    assert 'Direct HiCLASS background quantities' in terminal
    assert 'Hubble(z)' in terminal

    markdown = io_module._print_info(
        {}, {}, markdown=True, background=background)
    capsys.readouterr()
    assert '## Direct HiCLASS Background Quantities' in markdown
    assert '`Hubble(z)`' in markdown


def test_print_info_includes_held_out_validation(capsys):
    class SimpleSpectrum:
        name = 'pk_m'
        k_min = 0.001
        k_max = 50.0
        z_min = 0.0
        z_max = 10.0
        ell_min = None
        ell_max = None

    class SimpleParams:
        _required = ['h']
        _ranges_by_region = {
            'thin': {'z_pk': [0.0, 2.0], 'h': [0.65, 0.73]},
            'std': {'z_pk': [0.0, 3.0], 'h': [0.6, 0.8]},
            'ext': {'z_pk': [0.0, 10.0], 'h': [0.5, 0.9]},
        }
        _derived = {'h': ['h']}

    validation = {
        'schema_version': 1,
        'model': 'lcdm',
        'thresholds_percent': [0.01, 0.05, 0.1, 1.0],
        'region_membership': 'cumulative_source_datasets',
        'splits': {},
        'results': [{
            'observable': 'pk_m',
            'method': 'direct',
            'region': 'std',
            'metric': 'relative_rms',
            'samples_valid': 20000,
            'percent_within': {
                '0.01': 98.5,
                '0.05': 99.5,
                '0.1': 99.9,
                '1.0': 100.0,
            },
        }],
    }
    spectra = {'pk_m': SimpleSpectrum()}
    params = {'pk_m': SimpleParams()}

    content = io_module._print_info(
        spectra, params, markdown=True, validation=validation)
    assert '## Held-out Test Accuracy' in content
    assert '| direct | std | 20,000 | 98.500% |' in content
    assert 'relative RMS error across output modes' in content

    capsys.readouterr()
    io_module._print_info(spectra, params, validation=validation)
    output = capsys.readouterr().out
    assert 'Held-out test accuracy' not in output

    io_module._print_info(
        spectra, params, name='pk_m', bounds='std',
        validation=validation)
    output = capsys.readouterr().out
    assert 'Held-out test accuracy' in output
    assert '20,000' in output


def test_optional_validation_report_loading(tmp_path):
    assert io_module._load_validation_report(
        'lcdm', root=str(tmp_path)) is None

    bundle = tmp_path / 'lcdm'
    bundle.mkdir()
    report = {
        'schema_version': 1,
        'model': 'lcdm',
        'thresholds_percent': [0.01],
        'results': [],
    }
    (bundle / 'validation.json').write_text(json.dumps(report))

    assert io_module._load_validation_report(
        'lcdm', root=str(tmp_path)) == report

    report['model'] = 'other'
    (bundle / 'validation.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='does not match'):
        io_module._load_validation_report('lcdm', root=str(tmp_path))


def test_print_info_rejects_unknown_trust_region():
    with pytest.raises(ValueError, match='bounds must be one of'):
        io_module._print_info({}, {}, bounds='wide')


def test_print_info_rejects_output_without_markdown():
    with pytest.raises(ValueError, match='markdown=True'):
        io_module._print_info({}, {}, output='README.md')


@pytest.mark.parametrize('name, expected_call', [
    ('cl_TT_lensed', 'lensed'),
    ('cl_TT', 'raw'),
])
def test_cell_class_source_follows_emulator_name(
        monkeypatch, name, expected_call):
    calls = []

    class FakeHiClass:
        def set(self, params):
            pass

        def compute(self):
            pass

        def lensed_cl(self, lmax):
            calls.append('lensed')
            return {'tt': np.ones(lmax + 1)}

        def raw_cl(self, lmax):
            calls.append('raw')
            return {'tt': np.ones(lmax + 1)}

    monkeypatch.setattr(spectra_module.hiclassy, 'HiClass', FakeHiClass)
    spectrum = spectra_module.Cell.__new__(spectra_module.Cell)
    spectrum.name = name
    spectrum.class_args = {}
    spectrum.class_high_prec = {}
    spectrum._get_input_params_class = (
        lambda params, precision, class_args, verbose=False: params)

    spectrum.get_from_class([2, 3], {})

    assert calls == [expected_call]


@pytest.mark.parametrize('method, kwargs', [
    ('get_pk', {'name': 'm'}),
    ('get_fk', {'name': 'm', 'get_from_pk': False}),
    ('get_fk', {'name': 'm', 'get_from_pk': True}),
])
def test_internal_chunking_preserves_values_and_order(hifast, method, kwargs):
    params = np.arange(14, dtype=float).reshape(7, 2)
    z = np.linspace(0.0, 1.0, 7)
    k = np.array([0.1, 0.2, 0.3])
    function = getattr(hifast, method)

    complete = function(k, z, params, batch_size=None, **kwargs)
    chunked = function(k, z, params, batch_size=3, **kwargs)
    np.testing.assert_allclose(chunked, complete)


def test_cell_internal_chunking_preserves_values_and_order(hifast):
    params = np.arange(14, dtype=float).reshape(7, 2)
    ell = np.array([2, 10, 20])
    complete = hifast.get_cell(ell, params, name='TT')
    chunked = hifast.get_cell(ell, params, name='TT', batch_size=3)
    np.testing.assert_allclose(chunked, complete)


def test_one_dimensional_array_is_one_cosmology(hifast):
    output = hifast.get_pk(
        [0.1, 0.2], 0.5, np.array([1.0, 2.0]), name='m')
    assert output.shape == (1, 1, 2)


def test_squeeze_removes_singleton_batch_dimension(hifast):
    output = hifast.get_pk(
        [0.1, 0.2], 0.5, {'a': 1.0, 'b': 2.0},
        name='m', squeeze=True)
    assert output.shape == (2,)


@pytest.mark.parametrize('squeeze, expected_shape', [
    (False, (1, 2, 3)),
    (True, (2, 3)),
])
def test_pk_and_class_have_matching_shapes(hifast, squeeze, expected_shape):
    k = np.array([0.1, 0.2, 0.3])
    z = np.array([0.0, 0.5])
    params = {'a': 1.0, 'b': 2.0}
    emulator = hifast.get_pk(k, z, params, name='m', squeeze=squeeze)
    from_class = hifast.get_pk_from_class(
        k, z, params, name='m', squeeze=squeeze)

    assert np.asarray(emulator).shape == expected_shape
    assert np.asarray(from_class).shape == expected_shape


@pytest.mark.parametrize('get_from_pk', [False, True])
@pytest.mark.parametrize('squeeze, expected_shape', [
    (False, (1, 2, 3)),
    (True, (2, 3)),
])
def test_fk_and_class_have_matching_shapes(
        hifast, get_from_pk, squeeze, expected_shape):
    k = np.array([0.1, 0.2, 0.3])
    z = np.array([0.0, 0.5])
    params = {'a': 1.0, 'b': 2.0}
    emulator = hifast.get_fk(
        k, z, params, name='m', get_from_pk=get_from_pk,
        squeeze=squeeze)
    from_class = hifast.get_fk_from_class(
        k, z, params, name='m', squeeze=squeeze)

    assert np.asarray(emulator).shape == expected_shape
    assert np.asarray(from_class).shape == expected_shape


@pytest.mark.parametrize('squeeze, expected_shape', [
    (False, (1, 3)),
    (True, (3,)),
])
def test_cell_and_class_have_matching_shapes(
        hifast, squeeze, expected_shape):
    ell = np.array([2, 10, 20])
    params = {'a': 1.0, 'b': 2.0}
    emulator = hifast.get_cell(
        ell, params, name='TT', squeeze=squeeze)
    from_class = hifast.get_cell_from_class(
        ell, params, name='TT', squeeze=squeeze)

    assert np.asarray(emulator).shape == expected_shape
    assert np.asarray(from_class).shape == expected_shape


def test_rejects_wrong_array_width(hifast):
    with pytest.raises(ValueError, match=r'Expected 2 parameter columns'):
        hifast.get_pk([0.1], 0.0, np.ones(3), name='m')


@pytest.mark.parametrize('batch_size', [0, -1, 1.5])
def test_rejects_invalid_batch_size(hifast, batch_size):
    with pytest.raises(ValueError, match='batch_size'):
        hifast.get_pk(
            [0.1], 0.0, {'a': 1.0, 'b': 2.0}, name='m',
            batch_size=batch_size)


@pytest.mark.parametrize('method, kwargs, offset', [
    ('get_pk', {'name': 'm'}, 0.0),
    ('get_fk', {'name': 'm', 'get_from_pk': False}, 20.0),
    ('get_fk', {'name': 'm', 'get_from_pk': True}, 10.0),
])
def test_pk_and_fk_evaluate_full_cosmology_redshift_grid(
        hifast, method, kwargs, offset):
    k = np.array([0.1, 0.2])
    z = np.array([0.0, 0.5, 1.0])
    params = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = getattr(hifast, method)(k, z, params, **kwargs)
    expected = np.asarray([
        [row[0] + 2*row[1] + z_one + offset + k for z_one in z]
        for row in params
    ])

    assert result.shape == (2, 3, 2)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('method, kwargs, offset', [
    ('get_pk', {'name': 'm'}, 0.0),
    ('get_fk', {'name': 'm', 'get_from_pk': False}, 20.0),
    ('get_fk', {'name': 'm', 'get_from_pk': True}, 10.0),
])
def test_pk_and_fk_paired_mode_evaluates_corresponding_rows(
        hifast, method, kwargs, offset):
    k = np.array([0.1, 0.2])
    z = np.array([0.0, 0.5, 1.0])
    params = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    function = getattr(hifast, method)

    complete = function(
        k, z, params, paired=True, batch_size=None, **kwargs)
    chunked = function(
        k, z, params, paired=True, batch_size=2, **kwargs)
    expected = np.asarray([
        row[0] + 2*row[1] + z_one + offset + k
        for row, z_one in zip(params, z)
    ])

    assert complete.shape == (3, 2)
    np.testing.assert_allclose(complete, expected)
    np.testing.assert_allclose(chunked, expected)


@pytest.mark.parametrize('method, kwargs', [
    ('get_pk', {'name': 'm'}),
    ('get_fk', {'name': 'm', 'get_from_pk': False}),
    ('get_fk', {'name': 'm', 'get_from_pk': True}),
])
def test_paired_mode_rejects_mismatched_lengths(hifast, method, kwargs):
    with pytest.raises(ValueError, match='equal lengths'):
        getattr(hifast, method)(
            [0.1], [0.0, 1.0],
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            paired=True, **kwargs)


def test_unknown_spectrum_name_is_informative(hifast):
    with pytest.raises(ValueError, match='not available'):
        hifast.get_params_names('pk_unknown')
