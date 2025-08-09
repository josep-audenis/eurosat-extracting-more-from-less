import datetime
import os
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

labels = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",\
          "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

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
