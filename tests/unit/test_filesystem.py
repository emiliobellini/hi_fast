"""Round-trip tests for filesystem, FITS, and emulator metadata helpers."""

import json

import joblib
import numpy as np
import pytest

from hi_fast._filesystem import (
    EmuFile, FitsFile, Folder, _load_validation_report)


def test_folder_creation_listing_filtering_and_subfolders(tmp_path):
    root = Folder(tmp_path / 'root').create()
    first = tmp_path / 'root' / 'first.txt'
    second = tmp_path / 'root' / 'second.dat'
    first.write_text('first')
    second.write_text('second')
    (tmp_path / 'root' / 'child').mkdir()

    assert not root.is_empty()
    assert root.list_files(patterns=r'.*\.txt$', unique=True) == [str(first)]
    assert root.list_subfolders(unique=True) == [
        str(tmp_path / 'root' / 'child')]
    assert root.subfolder('child', should_exist=True).exists
    assert root.join('first.txt') == str(first)
    with pytest.raises(Exception, match='Multiple files'):
        root.list_files(unique=True)


def test_missing_folder_and_empty_folder_behaviour(tmp_path):
    missing = Folder(tmp_path / 'missing')
    assert missing.is_empty()
    assert missing.list_files() == []
    with pytest.raises(IOError, match='does not exist'):
        Folder(tmp_path / 'missing', should_exist=True)


def test_fits_data_and_nested_header_round_trip(tmp_path):
    path = tmp_path / 'nested' / 'sample.fits'
    fits_file = FitsFile(path)
    header = {
        'cosmology': {
            'name': 'lcdm',
            'values': [['1.0', '2.0'], ['3.0', '4.0']],
        },
        'labels': ['matter', 'cmb'],
    }
    primary = np.arange(6).reshape(2, 3)
    extension = np.array([10.0, 20.0])

    fits_file.write(data=primary, header=header)
    fits_file.write(name='EXTRA', data=extension)

    np.testing.assert_array_equal(fits_file.get_data('PRIMARY'), primary)
    np.testing.assert_array_equal(fits_file.get_data('EXTRA'), extension)
    assert fits_file.get_keys() == ['PRIMARY', 'EXTRA']
    restored = fits_file.get_header('PRIMARY')
    assert restored['cosmology']['name'] == 'lcdm'
    assert restored['cosmology']['values'] == [[1.0, 2.0], [3.0, 4.0]]
    assert restored['labels'] == ['matter', 'cmb']

    fits_file.update('EXTRA', data=np.array([30.0]))
    np.testing.assert_array_equal(fits_file.get_data('EXTRA'), [30.0])


def test_fits_file_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match='Expected .fits file'):
        FitsFile(tmp_path / 'sample.txt')


def test_emulator_file_detects_and_loads_dictionary(tmp_path):
    path = tmp_path / 'emulator.joblib'
    joblib.dump({'name': 'pk_m', 'value': 3}, path)

    emulator = EmuFile(path, should_exist=True)
    assert emulator.exists
    assert emulator._is_dict_file()
    assert emulator.load() == {'name': 'pk_m', 'value': 3}

    non_dict = tmp_path / 'array.joblib'
    joblib.dump(np.arange(3), non_dict)
    assert not EmuFile(non_dict)._is_dict_file()

    relative = EmuFile('emulator.joblib')
    assert relative.load(
        fname='emulator.joblib', root=Folder(tmp_path)) == {
        'name': 'pk_m', 'value': 3}


def test_emulator_file_missing_path_and_invalid_roots(tmp_path):
    with pytest.raises(IOError, match='does not exist'):
        EmuFile(tmp_path / 'missing.joblib', should_exist=True)
    with pytest.raises(ValueError, match='root not recognized'):
        EmuFile('sample.joblib', root=object())


def test_validation_report_loading_and_schema_checks(tmp_path):
    model_dir = tmp_path / 'lcdm'
    model_dir.mkdir()
    path = model_dir / 'validation.json'
    report = {'schema_version': 1, 'model': 'lcdm', 'results': []}
    path.write_text(json.dumps(report))

    assert _load_validation_report('lcdm', root=tmp_path) == report
    assert _load_validation_report(None, root=tmp_path) is None
    assert _load_validation_report('missing', root=tmp_path) is None

    path.write_text(json.dumps({**report, 'schema_version': 2}))
    with pytest.raises(ValueError, match='Unsupported validation'):
        _load_validation_report('lcdm', root=tmp_path)
    path.write_text(json.dumps({**report, 'model': 'other'}))
    with pytest.raises(ValueError, match='does not match'):
        _load_validation_report('lcdm', root=tmp_path)
