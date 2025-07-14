import datetime
import os
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",\
          "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

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



def generate_confusion_matrix(confusion_matrix, labels, filepath):
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="rocket", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    #plt.title("Confussion_matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


    return 

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
            "f1":14,
            "confusion_matrix": [[912,   2,   5,  11,   0,  23,   6,   0,  28,  13],\
 [  0, 978,   6,   0,   0,   4,   0,   1,   3,   8],\
 [  9,   6, 876,  13,  12,   4,  46,  24,  10,   0],\
 [ 45,   2,  29, 532,  38,  18,  57,  15,  97,   1],\
 [  0,   0,   2,  33, 774,   0,   6,  18,   1,   0],\
 [ 10,   3,  25,  13,   0, 595,  10,   0,   8,   2],\
 [ 22,   0,  51,  38,  18,  10, 686,   2,   6,   0],\
 [  0,   0,   2,   6,  12,   0,   1, 979,   0,   0],\
 [ 32,   7,   9,  86,  10,  22,   9,   0, 656,   2],\
 [  9,   3,   1,   0,   0,   3,   0,   0,  14, 924]]
        },
        {
            "accuracy":21,
            "precision":22,
            "recall":23,
            "f1":24,
            "confusion_matrix": [[912,   2,   5,  11,   0,  23,   6,   0,  28,  13],\
 [  0, 978,   6,   0,   0,   4,   0,   1,   3,   8],\
 [  9,   6, 876,  13,  12,   4,  46,  24,  10,   0],\
 [ 45,   2,  29, 532,  38,  18,  57,  15,  97,   1],\
 [  0,   0,   2,  33, 774,   0,   6,  18,   1,   0],\
 [ 10,   3,  25,  13,   0, 595,  10,   0,   8,   2],\
 [ 22,   0,  51,  38,  18,  10, 686,   2,   6,   0],\
 [  0,   0,   2,   6,  12,   0,   1, 979,   0,   0],\
 [ 32,   7,   9,  86,  10,  22,   9,   0, 656,   2],\
 [  9,   3,   1,   0,   0,   3,   0,   0,  14, 924]]

        },
        {
            "accuracy":31,
            "precision":32,
            "recall":33,
            "f1":34,
            "confusion_matrix": [[912,   2,   5,  11,   0,  23,   6,   0,  28,  13],\
 [  0, 978,   6,   0,   0,   4,   0,   1,   3,   8],\
 [  9,   6, 876,  13,  12,   4,  46,  24,  10,   0],\
 [ 45,   2,  29, 532,  38,  18,  57,  15,  97,   1],\
 [  0,   0,   2,  33, 774,   0,   6,  18,   1,   0],\
 [ 10,   3,  25,  13,   0, 595,  10,   0,   8,   2],\
 [ 22,   0,  51,  38,  18,  10, 686,   2,   6,   0],\
 [  0,   0,   2,   6,  12,   0,   1, 979,   0,   0],\
 [ 32,   7,   9,  86,  10,  22,   9,   0, 656,   2],\
 [  9,   3,   1,   0,   0,   3,   0,   0,  14, 924]]
        }
    ]

    generate_cross_validation_report(results=results, model_name="random forest", fold_metrics=fold_metrics)
