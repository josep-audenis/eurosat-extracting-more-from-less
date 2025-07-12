import datetime
import os
import subprocess

def generate_cross_validation_report(results, model_name, fold_metrics=[]):

    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report_path = "./docs/reports/cross_validation/cross_validation_report_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    
    n_folds = len(fold_metrics)

    template = open("./docs/reports/cross_validation/cross_validation_template.tex").read()

    fold_sections = generate_kfolds_tables(fold_metrics, n_folds)

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
        .replace('{{FOLD_SECTIONS}}', fold_sections)

    tex_file = ensure_tex_filename(report_path)
    with open(tex_file, 'w') as f:
        f.write(tex_filled)

    print(f"Latex report generated at {tex_file}")

    compile_latex(tex_file)

    return


def ensure_tex_filename(report_path):
    base, ext = os.path.splitext(report_path)
    if ext.lower() == '.pdf' or ext.lower() == '.tex':
        return base + '.tex'
    else:
        return report_path + '.tex'


def compile_latex(tex_path, clean_aux_files=True):
    tex_dir = os.path.dirname(tex_path) or '.'
    tex_file = os.path.basename(tex_path)
    base_name = os.path.splitext(tex_file)[0]

    try:
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_file],
            cwd=tex_dir,
            check=True,
            #stdout=subprocess.DEVNULL,
            #stderr=subprocess.DEVNULL
        )
        print(f"PDF compiled successfully: {base_name}.pdf")
    except subprocess.CalledProcessError:
        print("LaTeX compilation failed.")

    if clean_aux_files:
        for ext in ['.aux', '.log']:
            try:
                os.remove(os.path.join(tex_dir, base_name + ext))
            except FileNotFoundError:
                pass  
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
            fold_sections += "\\end{tabularx}"

    if total % 2 == 1:
        fold_sections += "\n\\vspace{1em}\n"

    print(fold_sections)

    return fold_sections


if __name__ == "__main__":
    results = {
        "accuracy_mean": 1,
        "accuracy_std": 2,
        "precision_mean": 3,
        "precision_std": 4,
        "recall_mean": 5,
        "recall_std": 6,
        "f1_mean": 7,
        "f1_std": 8
    }

    fold_metrics = [
        {
            "accuracy":11,
            "precision":12,
            "recall":13,
            "f1":14
        },
        {
            "accuracy":21,
            "precision":22,
            "recall":23,
            "f1":24
        },
        {
            "accuracy":31,
            "precision":32,
            "recall":33,
            "f1":34
        }
    ]

    generate_cross_validation_report(results=results, model_name="random forest", fold_metrics=fold_metrics)
