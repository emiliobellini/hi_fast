"""Render emulator metadata for terminals and Markdown files."""

import os

from tabulate import tabulate

from ._terminal import (
    info, print_level, write_blue, write_green, write_magenta,
)

_INFO_BOUNDS = ('thin', 'std', 'ext')


def _format_info_value(value):
    """Format range values compactly for terminal tables."""
    if isinstance(value, float):
        return '{:.8g}'.format(value)
    return value


def _format_info_range(bounds):
    """Return a compact ``[min, max]`` range string."""
    if bounds is None:
        return 'N/A'
    low, high = bounds
    return '[{}, {}]'.format(
        _format_info_value(low), _format_info_value(high))


def _spectrum_group(spec_name):
    """Return a display group for a spectrum name."""
    if spec_name.startswith('pk_'):
        return 'Power spectra'
    if spec_name.startswith('fk_'):
        return 'Growth rates'
    if spec_name.startswith('cl_'):
        return 'CMB spectra'
    return 'Other spectra'


def _spectrum_public_call(spec_name):
    """Return the public getter syntax for a spectrum name."""
    if spec_name.startswith('pk_'):
        name = spec_name.removeprefix('pk_')
        return 'get_pk(..., name="{}")'.format(name)
    if spec_name.startswith('fk_'):
        name = spec_name.removeprefix('fk_')
        return 'get_fk(..., name="{}")'.format(name)
    if spec_name.startswith('cl_'):
        name = spec_name.removeprefix('cl_').removesuffix('_lensed')
        return 'get_cell(..., name="{}")'.format(name)
    return spec_name


def _build_info_metadata(spectra, params, name=None, validation=None):
    """Build structured metadata shared by terminal and Markdown renderers."""
    if name is None:
        spec_names = sorted(spectra)
    else:
        spec_names = [name]

    metadata = []
    validation_results = {} if validation is None else {
        spec_name: [
            result for result in validation.get('results', [])
            if result.get('observable') == spec_name
        ]
        for spec_name in spec_names
    }
    for spec_name in spec_names:
        spec = spectra[spec_name]
        param = params[spec_name]
        entry = {
            'name': spec_name,
            'group': _spectrum_group(spec_name),
            'public_call': _spectrum_public_call(spec_name),
            'required': list(param._required),
            'derived': {
                p_name: [
                    x for x in param._derived[p_name] if x != p_name
                ]
                for p_name in param._required
            },
            'ranges': {
                region: {
                    p_name: param._ranges_by_region[region].get(p_name)
                    for p_name in param._required
                }
                for region in _INFO_BOUNDS
            },
            'z_ranges': {
                region: param._ranges_by_region[region].get('z_pk')
                for region in _INFO_BOUNDS
            },
            'k_range': None,
            'ell_range': None,
            'validation': validation_results.get(spec_name, []),
            'validation_thresholds': (
                [] if validation is None
                else validation.get('thresholds_percent', [])),
            'validation_split': (
                None if validation is None
                else validation.get('splits', {}).get(spec_name)),
            'validation_region_membership': (
                None if validation is None
                else validation.get('region_membership')),
        }
        if spec.k_min is not None and spec.k_max is not None:
            entry['k_range'] = [spec.k_min, spec.k_max]
        if spec.ell_min is not None and spec.ell_max is not None:
            entry['ell_range'] = [spec.ell_min, spec.ell_max]
        metadata.append(entry)
    return metadata


def _info_domain_summary(entry, bounds):
    """Return compact grid/domain text from metadata."""
    entries = []
    if entry['k_range'] is not None:
        entries.append('k {}'.format(_format_info_range(entry['k_range'])))
    if entry['ell_range'] is not None:
        entries.append('ell {}'.format(
            _format_info_range(entry['ell_range'])))
    if entry['z_ranges']['ext'] is not None:
        if bounds is None:
            z_ranges = [
                '{} {}'.format(
                    region, _format_info_range(entry['z_ranges'][region]))
                for region in _INFO_BOUNDS
            ]
            entries.append('z: {}'.format('; '.join(z_ranges)))
        else:
            entries.append('z {}'.format(
                _format_info_range(entry['z_ranges'][bounds])))
    return '; '.join(entries) if entries else 'N/A'


def _info_detail_rows(entry, bounds, color=False):
    """Return detailed parameter/domain rows for one metadata entry."""
    rows = []
    for p_name in entry['required']:
        der = ', '.join(entry['derived'][p_name])
        display_name = write_blue(p_name) if color else p_name
        if bounds is None:
            row = [display_name]
            for region in _INFO_BOUNDS:
                row.append(_format_info_range(
                    entry['ranges'][region].get(p_name)))
            row.append(der)
        else:
            p_range = entry['ranges'][bounds].get(p_name)
            if p_range is None:
                p_min, p_max = 'N/A', 'N/A'
            else:
                p_min, p_max = [_format_info_value(x) for x in p_range]
            row = [display_name, p_min, p_max, der]
        rows.append(row)

    if entry['z_ranges']['ext'] is not None:
        display_name = write_magenta('z') if color else 'z'
        if bounds is None:
            row = [display_name]
            for region in _INFO_BOUNDS:
                row.append(_format_info_range(entry['z_ranges'][region]))
            row.append('N/A')
        else:
            z_min, z_max = entry['z_ranges'][bounds]
            row = [display_name, _format_info_value(z_min),
                   _format_info_value(z_max), 'N/A']
        rows.append(row)

    if entry['k_range'] is not None:
        display_name = write_magenta('k [h/Mpc]') if color else 'k [h/Mpc]'
        if bounds is None:
            k_range = _format_info_range(entry['k_range'])
            rows.append([display_name, k_range, k_range, k_range, 'N/A'])
        else:
            rows.append([display_name,
                         _format_info_value(entry['k_range'][0]),
                         _format_info_value(entry['k_range'][1]), 'N/A'])

    if entry['ell_range'] is not None:
        display_name = write_magenta('ell') if color else 'ell'
        if bounds is None:
            ell_range = _format_info_range(entry['ell_range'])
            rows.append([display_name, ell_range, ell_range, ell_range, 'N/A'])
        else:
            rows.append([display_name,
                         _format_info_value(entry['ell_range'][0]),
                         _format_info_value(entry['ell_range'][1]), 'N/A'])

    return rows


def _markdown_cell(value):
    """Escape a value for use in a Markdown table cell."""
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _markdown_table(headers, rows):
    """Return a GitHub-flavored Markdown table."""
    lines = [
        '| {} |'.format(' | '.join(_markdown_cell(x) for x in headers)),
        '| {} |'.format(' | '.join('---' for _ in headers)),
    ]
    for row in rows:
        lines.append('| {} |'.format(
            ' | '.join(_markdown_cell(x) for x in row)))
    return '\n'.join(lines)


def _validation_rows(entry, bounds=None):
    """Return sorted held-out validation records for one observable."""
    region_order = {region: index for index, region in enumerate(_INFO_BOUNDS)}
    method_order = {'direct': 0, 'from_pk': 1}
    results = [
        result for result in entry['validation']
        if bounds is None or result.get('region') == bounds
    ]
    return sorted(
        results,
        key=lambda result: (
            region_order.get(result.get('region'), len(region_order)),
            method_order.get(result.get('method'), len(method_order))))


def _validation_threshold_headers(entry, metric):
    """Format relative or absolute RMS threshold headings."""
    if metric == 'relative_rms':
        return ['≤{}%'.format(value)
                for value in entry['validation_thresholds']]
    return ['≤{:g}'.format(value / 100.0)
            for value in entry['validation_thresholds']]


def _validation_table(entry, bounds=None):
    """Return headers and rows for one validation report table."""
    results = _validation_rows(entry, bounds=bounds)
    if not results:
        return None, []
    metric = results[0]['metric']
    headers = ['Method', 'Region', 'Test models']
    headers += _validation_threshold_headers(entry, metric)
    rows = []
    for result in results:
        percentages = result['percent_within']
        rows.append([
            result['method'],
            result['region'],
            '{:,}'.format(result['samples_valid']),
        ] + [
            ('{:.3f}%'.format(percentages[str(threshold)])
             if percentages[str(threshold)] is not None else 'N/A')
            for threshold in entry['validation_thresholds']
        ])
    return headers, rows


def _validation_metric_description(metric):
    """Return a human-readable per-sample RMS metric description."""
    if metric == 'relative_rms':
        return 'relative RMS error across output modes'
    return 'absolute RMS error across output modes'


def _append_validation_markdown(lines, metadata, bounds):
    """Append held-out validation tables when a report is available."""
    entries = [entry for entry in metadata if _validation_rows(entry, bounds)]
    if not entries:
        return
    lines.extend([
        '## Held-out Test Accuracy',
        '',
        ('The original global train/test split is reconstructed from the '
         'stored training fraction, random seed, finite-row filtering, and '
         'dataset order. Region results are cumulative by source dataset: '
         '`std` combines the `thin` and `std` test rows, while `ext` combines '
         'all three source regions.'),
        '',
    ])
    for entry in entries:
        results = _validation_rows(entry, bounds)
        metric = results[0]['metric']
        headers, rows = _validation_table(entry, bounds=bounds)
        lines.extend([
            '### {}'.format(entry['name']),
            '',
            'Metric: {}.'.format(_validation_metric_description(metric)),
            '',
            _markdown_table(headers, rows),
            '',
        ])


def _print_validation(metadata, bounds):
    """Print held-out validation tables when a report is available."""
    entries = [entry for entry in metadata if _validation_rows(entry, bounds)]
    if not entries:
        return
    info('Held-out test accuracy (cumulative source regions):')
    for entry in entries:
        results = _validation_rows(entry, bounds)
        metric = results[0]['metric']
        print_level(1, '{} — {}'.format(
            entry['name'], _validation_metric_description(metric)))
        headers, rows = _validation_table(entry, bounds=bounds)
        print(tabulate(rows, headers=headers, tablefmt='grid'))


def _format_info_markdown(
        metadata, name=None, bounds=None, background=None):
    """Render emulator metadata as Markdown."""
    lines = []
    if name is None:
        lines.append('# HiFast Emulator Summary')
    else:
        lines.append('# HiFast Emulator: {}'.format(name))
    lines.append('')
    if bounds is None:
        lines.append(
            'Trust regions are shown as `thin`, `std`, and `ext`.')
    else:
        lines.append('Trust region: `{}`.'.format(bounds))
    lines.append(
        'Select one with `trusted_region` in a public `get_*` call. '
        'Use `on_out_of_bounds="class"` for automatic HiCLASS fallback, '
        'or `trusted_region=None` to use HiCLASS for every input.')
    lines.append('')

    groups = ('Power spectra', 'Growth rates', 'CMB spectra', 'Other spectra')
    if name is None:
        lines.append('## Observables')
        for group in groups:
            group_entries = [
                entry for entry in metadata if entry['group'] == group
            ]
            if not group_entries:
                continue
            lines.append('')
            lines.append('### {}'.format(group))
            lines.append('')
            rows = []
            for entry in group_entries:
                rows.append([
                    entry['name'],
                    '`{}`'.format(entry['public_call']),
                    ', '.join('`{}`'.format(x) for x in entry['required']),
                    _info_domain_summary(entry, bounds),
                ])
            lines.append(_markdown_table(
                ['Observable', 'Public call', 'Required inputs',
                 'Grid / redshift domain'],
                rows))
            lines.append('')

        if background:
            lines.extend([
                '## Direct HiCLASS Background Quantities',
                '',
                ('These quantities are available through `get_background`; '
                 '`get_background_table` returns the complete native '
                 'HiCLASS table.'),
                '',
                _markdown_table(
                    ['HiFast name', 'HiCLASS member', 'Additional input',
                     'Units'],
                    [[entry['name'], '`{}`'.format(entry['hiclassy']),
                      entry['input'], entry['units']]
                     for entry in background]),
                '',
            ])

    _append_validation_markdown(lines, metadata, bounds)

    lines.append('## Detailed Trust Regions')
    for entry in metadata:
        lines.append('')
        lines.append('### {}'.format(entry['name']))
        lines.append('')
        if bounds is None:
            headers = ['Parameter', 'Thin', 'Std', 'Ext',
                       'Can be derived from']
        else:
            headers = ['Parameter', 'Min', 'Max', 'Can be derived from']
        rows = _info_detail_rows(entry, bounds, color=False)
        rows = [
            [row[0]] + row[1:-1] + [
                ', '.join('`{}`'.format(x) for x in row[-1].split(', '))
                if row[-1] not in ('', 'N/A') else row[-1]
            ]
            for row in rows
        ]
        lines.append(_markdown_table(headers, rows))
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def _print_summary(metadata, bounds, background=None):
    """Print a compact grouped overview of all loaded observables."""
    info('HiFast emulator summary:')
    if bounds is None:
        info('Trust regions shown as thin / std / ext. Use '
             'print_info(name, bounds="std") for one selected region.')
    else:
        info('Showing trust region: {}'.format(bounds))
    info('Choose one with trusted_region; set on_out_of_bounds="class" '
         'for HiCLASS fallback.')

    groups = ('Power spectra', 'Growth rates', 'CMB spectra', 'Other spectra')
    for group in groups:
        group_entries = [
            entry for entry in metadata if entry['group'] == group
        ]
        if not group_entries:
            continue
        print('\n')
        print_level(1, group)
        headers = ['Observable', 'Public call', 'Required inputs',
                   'Grid / redshift domain']
        headers = [write_green(x) for x in headers]
        tab = []
        for entry in group_entries:
            tab.append([
                write_blue(entry['name']),
                entry['public_call'],
                ', '.join(entry['required']),
                _info_domain_summary(entry, bounds),
            ])
        print(tabulate(tab, headers=headers, tablefmt='grid'))
    if background:
        print('\n')
        print_level(1, 'Direct HiCLASS background quantities')
        headers = ['HiFast name', 'HiCLASS member', 'Additional input',
                   'Units']
        headers = [write_green(x) for x in headers]
        rows = [[write_blue(entry['name']), entry['hiclassy'],
                 entry['input'], entry['units']]
                for entry in background]
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        info('Use get_background(...) for selected values or '
             'get_background_table(...) for the native HiCLASS table.')
    return


def _print_detail(metadata, bounds):
    """Print detailed information for a single spectrum emulator."""
    info('HiFast emulator info:')
    info('Choose a region with trusted_region; set '
         'on_out_of_bounds="class" for HiCLASS fallback.')

    for entry in metadata:
        print('\n')
        print_level(1, 'Spectrum: {}'.format(entry['name']))

        if bounds is None:
            headers = ['Parameter', 'Thin', 'Std', 'Ext',
                       'Can be derived from']
        else:
            headers = ['Parameter', 'Min', 'Max', 'Can be derived from']
        headers = [write_green(x) for x in headers]
        tab = _info_detail_rows(entry, bounds, color=True)
        print(tabulate(tab, headers=headers, tablefmt='grid'))

    _print_validation(metadata, bounds)
    return


def _print_info(
        spectra, params, name=None, bounds=None, markdown=False, output=None,
        validation=None, background=None):
    """Print summary or detailed info for spectrum emulators.
    Args:
        spectra (dict): Mapping from spectrum names to Spectrum objects.
        params (dict): Mapping from spectrum names to Params objects.
        name (str | None): When provided, print info only for the named
            spectrum.
        bounds (str | None): Optional trust region to display. Accepted
            values are ``thin``, ``std``, and ``ext``. When omitted, all
            available regions are shown.
        markdown (bool): When True, render Markdown instead of terminal
            tables.
        output (str | None): Optional file path used only with
            ``markdown=True``. When omitted, Markdown is printed to stdout.
        validation (dict | None): Optional held-out validation report.
        background (list[dict] | None): Public background nomenclature shown
            in the all-observables summary.
    """
    if bounds is not None and bounds not in _INFO_BOUNDS:
        raise ValueError('bounds must be one of {}; got {}'.format(
            _INFO_BOUNDS, bounds))
    if output is not None and markdown is False:
        raise ValueError('output can only be used with markdown=True')

    metadata = _build_info_metadata(
        spectra, params, name=name, validation=validation)
    if markdown:
        content = _format_info_markdown(
            metadata, name=name, bounds=bounds, background=background)
        if output is None:
            print(content, end='')
        else:
            parent = os.path.dirname(os.path.abspath(output))
            os.makedirs(parent, exist_ok=True)
            with open(output, 'w', encoding='utf-8') as file:
                file.write(content)
            info('Wrote emulator README to {}'.format(output))
        return content

    if name is None:
        return _print_summary(metadata, bounds, background=background)
    return _print_detail(metadata, bounds)


# ------------------- Scripts ------------------------------------------------#
