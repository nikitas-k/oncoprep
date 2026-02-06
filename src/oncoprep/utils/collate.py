"""Self-contained function for Nipype Function node to collate subject reports.

Produces an fMRIPrep-style XHTML report with Bootstrap 4 navbar.
"""


def collate_subject_report(output_dir, subject_id, version,
                           report_files, workflow_desc=''):
    """Assemble every reportlet under ``oncoprep/`` into one XHTML file.

    Parameters
    ----------
    output_dir : str
        Top-level derivatives directory (e.g. ``<bids>/derivatives``).
        Figures live under ``<output_dir>/oncoprep/<subject_id>/figures/``.
    subject_id : str
        BIDS subject label including ``sub-`` prefix.
    version : str
        OncoPrep version string.
    report_files : list
        Sentinel list used only for Nipype DAG ordering.
    workflow_desc : str
        Boilerplate workflow description text.

    Returns
    -------
    str
        Path to the written HTML report.
    """
    import re
    from pathlib import Path as _Path

    # DerivativesDataSink uses out_path_base='oncoprep'
    deriv_dir = _Path(output_dir) / 'oncoprep'
    figures_dir = deriv_dir / subject_id / 'figures'
    fig_rel = subject_id + '/figures'

    def _read(p):
        try:
            return _Path(p).read_text(encoding='utf-8')
        except Exception:
            return ''

    NORM_LABELS = {
        '_T1w.svg': ('T1w', 'Spatial normalization of the T1w image'),
        '_acq-ce_T1w.svg': ('T1ce', 'Spatial normalization of the T1ce image'),
        '_T2w.svg': ('T2w', 'Spatial normalization of the T2w image'),
        '_FLAIR.svg': ('FLAIR', 'Spatial normalization of the FLAIR image'),
    }

    # ---- collect reportlets ----
    summary_html = about_html = conform_html = ''
    dseg_svg_name = tumor_svg_name = ''
    norm_figures = []  # [(caption, filename), ...]

    if figures_dir.is_dir():
        for fpath in sorted(figures_dir.glob('*')):
            fname = fpath.name
            if fname.endswith('_desc-summary_T1w.html'):
                summary_html = _read(fpath)
            elif fname.endswith('_desc-about_T1w.html'):
                about_html = _read(fpath)
            elif fname.endswith('_desc-conform_T1w.html'):
                conform_html = _read(fpath)
            elif fname.endswith('_desc-tumor_dseg.svg'):
                tumor_svg_name = fname
            elif fname.endswith('_dseg.svg') and 'tumor' not in fname:
                dseg_svg_name = fname
            elif fname.endswith('.svg') and 'space-' in fname:
                caption = fname
                for suffix, (modality, desc) in NORM_LABELS.items():
                    if fname.endswith(suffix):
                        m = re.search(r'space-([A-Za-z0-9]+)', fname)
                        space = m.group(1) if m else 'standard'
                        caption = (
                            desc + ' to the '
                            '<code>' + space + '</code> template.'
                        )
                        break
                norm_figures.append((caption, fname))

    # ---- Assemble sections ----
    sections = []

    # -- Summary --
    sec = '    <div id="Summary">\n'
    sec += '    <h1 class="sub-report-title">Summary</h1>\n'
    if summary_html:
        sec += (
            '        <div id="datatype-figures_desc-summary_suffix-T1w">\n'
            '                    ' + summary_html.strip() + '\n'
            '        </div>\n'
        )
    sec += '    </div>\n'
    sections.append(sec)

    # -- Anatomical --
    sec = '    <div id="Anatomical">\n'
    sec += '    <h1 class="sub-report-title">Anatomical</h1>\n'
    if conform_html:
        sec += (
            '        <div id="datatype-figures_desc-conform_suffix-T1w">\n'
            '                    ' + conform_html.strip() + '\n'
            '        </div>\n'
        )
    if dseg_svg_name:
        sec += (
            '        <div id="datatype-figures_suffix-dseg">\n'
            '<h3 class="run-title">Brain mask and brain tissue segmentation'
            ' of the T1w</h3>'
            '<p class="elem-caption">This panel shows the template '
            'T1-weighted image, with contours delineating the detected '
            'brain mask and brain tissue segmentations.</p>'
            '                    <object class="svg-reportlet" '
            'type="image/svg+xml" data="./' + fig_rel + '/' + dseg_svg_name
            + '">\nProblem loading figure '
            + fig_rel + '/' + dseg_svg_name
            + '. If the link below works, please try reloading the report '
            'in your browser.</object>\n'
            '</div>\n'
            '<div class="elem-filename">\n'
            '    Get figure file: <a href="./' + fig_rel + '/'
            + dseg_svg_name + '" target="_blank">'
            + fig_rel + '/' + dseg_svg_name + '</a>\n'
            '</div>\n\n'
        )
    if norm_figures:
        sec += (
            '        <div id="datatype-figures_space_suffix-T1w">\n'
            '<h3 class="run-title">Spatial normalization of the anatomical '
            'T1w reference</h3>'
            '<p class="elem-desc">Results of nonlinear alignment of the T1w '
            'reference one or more template space(s). Hover on the panels '
            'with the mouse pointer to transition between both spaces.</p>'
        )
        for caption, fname in norm_figures:
            sec += (
                '<p class="elem-caption">' + caption + '</p>'
                '                    <object class="svg-reportlet" '
                'type="image/svg+xml" data="./' + fig_rel + '/' + fname
                + '">\nProblem loading figure '
                + fig_rel + '/' + fname
                + '. If the link below works, please try reloading the '
                'report in your browser.</object>\n'
                '</div>\n'
                '<div class="elem-filename">\n'
                '    Get figure file: <a href="./' + fig_rel + '/'
                + fname + '" target="_blank">'
                + fig_rel + '/' + fname + '</a>\n'
                '</div>\n\n'
            )
    sec += '    </div>\n'
    sections.append(sec)

    # -- Tumor Segmentation --
    if tumor_svg_name:
        sec = '    <div id="Segmentation">\n'
        sec += '    <h1 class="sub-report-title">Tumor Segmentation</h1>\n'
        sec += (
            '        <div id="datatype-figures_desc-tumor_suffix-dseg">\n'
            '<h3 class="run-title">Tumor region contour overlay</h3>'
            '<p class="elem-caption">Tumor segmentation contours overlaid '
            'on the T1-weighted reference image. Regions correspond to '
            'BraTS label classes: necrotic core (label 1), '
            'peritumoral edema (label 2), GD-enhancing tumor (label 4).'
            '</p>'
            '                    <object class="svg-reportlet" '
            'type="image/svg+xml" data="./' + fig_rel + '/'
            + tumor_svg_name + '">\nProblem loading figure '
            + fig_rel + '/' + tumor_svg_name
            + '. If the link below works, please try reloading the report '
            'in your browser.</object>\n'
            '</div>\n'
            '<div class="elem-filename">\n'
            '    Get figure file: <a href="./' + fig_rel + '/'
            + tumor_svg_name + '" target="_blank">'
            + fig_rel + '/' + tumor_svg_name + '</a>\n'
            '</div>\n\n'
            '    </div>\n'
        )
        sections.append(sec)

    # -- About --
    sec = '    <div id="About">\n'
    sec += '    <h1 class="sub-report-title">About</h1>\n'
    if about_html:
        sec += (
            '        <div id="datatype-figures_desc-about_suffix-T1w">\n'
            '                    ' + about_html.strip() + '\n'
            '        </div>\n'
        )
    sec += '    </div>\n'
    sections.append(sec)

    # -- Boilerplate / Methods --
    sec = '<div id="boilerplate">\n'
    sec += '    <h1 class="sub-report-title">Methods</h1>\n'
    if workflow_desc:
        sec += (
            '    <p>We kindly ask to report results preprocessed with '
            'this tool using the following boilerplate.</p>\n'
            '    <div class="boiler-html">\n'
            + workflow_desc.strip() + '\n'
            '    </div>\n'
        )
    sec += '</div>\n'
    sections.append(sec)

    # -- Errors --
    sec = (
        '<div id="errors">\n'
        '    <h1 class="sub-report-title">Errors</h1>\n'
        '    <p>No errors to report!</p>\n'
        '</div>\n'
    )
    sections.append(sec)

    body = '\n'.join(sections)

    # ---- build navbar links ----
    nav_items = (
        '<li class="nav-item">'
        '<a class="nav-link" href="#Summary">Summary</a></li>\n'
        '        <li class="nav-item">'
        '<a class="nav-link" href="#Anatomical">Anatomical</a></li>\n'
    )
    if tumor_svg_name:
        nav_items += (
            '        <li class="nav-item">'
            '<a class="nav-link" href="#Segmentation">'
            'Tumor Segmentation</a></li>\n'
        )
    nav_items += (
        '        <li class="nav-item">'
        '<a class="nav-link" href="#About">About</a></li>\n'
        '        <li class="nav-item">'
        '<a class="nav-link" href="#boilerplate">Methods</a></li>\n'
        '        <li class="nav-item">'
        '<a class="nav-link" href="#errors">Errors</a></li>\n'
    )

    html = (
        '<?xml version="1.0" encoding="utf-8" ?>\n'
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"'
        ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en"'
        ' lang="en">\n'
        '<head>\n'
        '<meta http-equiv="Content-Type"'
        ' content="text/html; charset=utf-8" />\n'
        '<meta name="generator" content="OncoPrep ' + version + '" />\n'
        '<title>OncoPrep - ' + subject_id + '</title>\n'
        '<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"'
        ' integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4Yf'
        'RvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>\n'
        '<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3'
        '/js/bootstrap.min.js" integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6'
        'ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy"'
        ' crossorigin="anonymous"></script>\n'
        '<link rel="stylesheet"'
        ' href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css'
        '/bootstrap.min.css" integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7'
        'Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO"'
        ' crossorigin="anonymous">\n'
        '<style type="text/css">\n'
        '.sub-report-title {}\n'
        '.run-title {}\n'
        '\n'
        'h1 { padding-top: 35px; }\n'
        'h2 { padding-top: 20px; }\n'
        'h3 { padding-top: 15px; }\n'
        '\n'
        '.elem-desc {}\n'
        '.elem-caption {\n'
        '    margin-top: 15px;\n'
        '    margin-bottom: 0;\n'
        '}\n'
        '.elem-filename {}\n'
        '\n'
        'div.elem-image {\n'
        '  width: 100%;\n'
        '  page-break-before:always;\n'
        '}\n'
        '\n'
        '.elem-image object.svg-reportlet {\n'
        '    width: 100%;\n'
        '    padding-bottom: 5px;\n'
        '}\n'
        'body {\n'
        '    padding: 65px 10px 10px;\n'
        '}\n'
        '\n'
        '.boiler-html {\n'
        '    font-family: "Bitstream Charter", "Georgia", Times;\n'
        '    margin: 20px 25px;\n'
        '    padding: 10px;\n'
        '    background-color: #F8F9FA;\n'
        '}\n'
        '\n'
        'div#boilerplate pre {\n'
        '    margin: 20px 25px;\n'
        '    padding: 10px;\n'
        '    background-color: #F8F9FA;\n'
        '}\n'
        '\n'
        '#errors div, #errors p {\n'
        '    padding-left: 1em;\n'
        '}\n'
        '\n'
        'object.svg-reportlet {\n'
        '    width: 100%;\n'
        '    padding-bottom: 5px;\n'
        '}\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
        '\n'
        '\n'
        '<nav class="navbar fixed-top navbar-expand-lg navbar-light'
        ' bg-light">\n'
        '<div class="collapse navbar-collapse">\n'
        '    <ul class="navbar-nav">\n'
        '        ' + nav_items
        + '    </ul>\n'
        '</div>\n'
        '</nav>\n'
        '<noscript>\n'
        '    <h1 class="text-danger"> The navigation menu uses Javascript.'
        ' Without it this report might not work as expected </h1>\n'
        '</noscript>\n'
        '\n'
        + body + '\n'
        '\n'
        '<script type="text/javascript">\n'
        '    function toggle(id) {\n'
        '        var element = document.getElementById(id);\n'
        "        if(element.style.display == 'block')\n"
        "            element.style.display = 'none';\n"
        '        else\n'
        "            element.style.display = 'block';\n"
        '    }\n'
        '</script>\n'
        '</body>\n'
        '</html>\n'
    )

    # Write to derivatives/oncoprep/sub-XXX.html
    report_path = deriv_dir / (subject_id + '.html')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding='utf-8')
    return str(report_path)
