import datetime

from report_utils import labels, ensure_tex_filename, compile_latex, generate_confusion_matrix



def generate_cross_validation_report(results, model_name, fold_metrics=[]):

    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report_path = "./docs/reports/cross_validation/cross_validation_report_" + datetime.datetime.now().strftime("%Y-%m-%d_%Hh-%Mm")
    
    n_folds = len(fold_metrics)

    template = open("./docs/reports/cross_validation/cross_validation_template.tex").read()

    fold_sections = generate_kfolds_tables(fold_metrics, n_folds)

    filenames = save_confusion_matrices_from_folds(fold_metrics, labels)

    cm_section = generate_confusion_matrix_section(filenames)

    tex_filled = template\
        .replace('{{DATE}}', today)\
        .replace('{{MODEL}}', model_name)\
        .replace('{{FOLDS}}', str(n_folds))\
        .replace('{{ACC_MEAN}}', f"{results['accuracy_mean']*100:.2f}")\
        .replace('{{ACC_STD}}', f"{results['accuracy_std']*100:.2f}")\
        .replace('{{PREC_MEAN}}', f"{results['precision_mean']*100:.2f}")\
        .replace('{{PREC_STD}}', f"{results['precision_std']*100:.2f}")\
        .replace('{{REC_MEAN}}', f"{results['recall_mean']*100:.2f}")\
        .replace('{{REC_STD}}', f"{results['recall_std']*100:.2f}")\
        .replace('{{F1_MEAN}}', f"{results['f1_mean']*100:.2f}")\
        .replace('{{F1_STD}}', f"{results['f1_std']*100:.2f}")\
        .replace('{{FOLD_SECTION}}', fold_sections)\
        .replace('{{CONFUSION_SECTION}}', cm_section)

    tex_file = ensure_tex_filename(report_path)
    with open(tex_file, 'w') as f:
        f.write(tex_filled)

    print(f"Latex report generated at {tex_file}")

    compile_latex(tex_file)

    return


def generate_kfolds_tables(results, n_folds):
    fold_sections = ""

    total = n_folds
    in_row = 0

    if total > 1:
        fold_sections += "\\begin{tabularx}{\\textwidth}{XX}\n"

    for i, fold_result in enumerate(results, 1):
        
        if total > 1 and in_row == 0 and i == total:
            fold_sections += "\n\\end{tabularx}"
        elif total > 1:
            fold_sections += "\n\\begin{minipage}[t]{\\linewidth}"

        table = f"""
\\begin{{center}}
    \\begin{{tabular}}{{lc}}
        \\hline
        \\textbf{{Metric}} & \\textbf{{Score (\\%)}}\\\\
        \\hline
        Accuracy & {fold_result['accuracy']*100:.2f}\\\\
        Precision & {fold_result['precision']*100:.2f}\\\\
        Recall & {fold_result['recall']*100:.2f}\\\\
        F1 Score & {fold_result['f1']*100:.2f}\\\\
        \\hline
    \\end{{tabular}}
    \\captionof{{table}}{{Performance metrics for Fold {i}}}
\\end{{center}}
"""

        fold_sections += table
        in_row += 1

        if i != total:
            fold_sections += "\\end{minipage}\n"
            if in_row == 1:
                fold_sections += "\n&\n"
            else:
                in_row = 0
        elif in_row == 1:
            fold_sections += ""
        elif in_row == 2:
            fold_sections += "\\end{minipage}\n\\end{tabularx}"

    if total % 2 == 1:
        fold_sections += "\n\\vspace{1em}\n"

    return fold_sections



def generate_confusion_matrix_section(filenames):
    section = ""

    total = len(filenames)
    in_row = 0
    image_width = 0.5

    if total > 1:
        section += "\\begin{tabularx}{\\textwidth}{XX}\n"

    for i, filename in enumerate(filenames, 1):
        
        if total > 1 and in_row == 0 and i == total:
            section += "\n\\end{tabularx}\n"
            image_width = 0.5
        elif total > 1:
            image_width = 1
            section += "\n\\begin{minipage}[t]{\\linewidth}"
        

        figure = f"""
\\begin{{center}}
\\includegraphics[width={image_width}\\linewidth]{{{f"../figures/conf_matrix_fold{i}.png"}}}
\\captionof{{figure}}{{\\centering Cross Validation Confusion Matrix of Fold {i}}}
\\end{{center}}
"""
        section += figure
        in_row += 1

        if i != total:
            section += "\\end{minipage}\n"
            if in_row == 1:
                section += "\n&\n"
            else:
                in_row = 0
        elif in_row == 1:
            section += ""
        elif in_row == 2:
            section += "\\end{minipage}\n\\end{tabularx}\n"

    if total % 2 == 1:
        section += "\n\\vspace{1em}\n"

    return section

def save_confusion_matrices_from_folds(fold_metrics, labels, output_dir="./docs/reports/figures/"):
    filenames = []
    
    for i, fold in enumerate(fold_metrics, 1):
        cm = fold["confusion_matrix"]
        filename = f"{output_dir}conf_matrix_fold{i}.png"
        generate_confusion_matrix(cm, labels, filename)
        filename = f"../figures/conf_matrix_fold{i}.png"
        filenames.append(filename)
    
    return filenames